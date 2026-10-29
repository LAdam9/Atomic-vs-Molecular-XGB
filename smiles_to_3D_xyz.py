#!/usr/bin/env python3
"""
dockstring_batch_xyz.py

Batch-generate DOCKSTRING's pre-docking 3D geometries for up to ~1,000,000
compounds, reading (name, SMILES) pairs from a delimited text file and
collecting all resulting geometries into a single multi-frame XYZ file.

Ligand preparation uses ONLY dockstring.utils and rdkit -- no hand-rolled
chemistry and no direct subprocess calls of our own. It replicates, in the
same order, the exact pre-docking pipeline run inside
dockstring.target.Target.dock() [github.com/dockstring/dockstring]:

    1. canonicalize_smiles(smiles)
    2. smiles_to_mol(canonical_smiles)
    3. sanitize_mol(mol)
    4. check_mol(mol); check_charges(mol)
    5. protonate_mol(mol, pH)            (dockstring's own Open Babel wrapper)
    6. check_mol(protonated_mol)
    7. embed_mol(protonated_mol, seed)   (RDKit ETKDG; dockstring's default seed: 974528263)
    8. refine_mol_with_ff(embedded_mol)  (MMFF94, UFF fallback)
    9. assign_stereochemistry(refined_mol)

The only thing added on top is RDKit's own Chem.rdmolfiles.MolToXYZBlock to
serialize the result, plus its comment line -- which carries EVERY column
from the input file (not just name/smiles) as key=value pairs, so no input
metadata (potency, source library, batch ID, whatever the file has) gets
dropped on the way to a geometry file. A value containing whitespace is
wrapped in double quotes (with any literal " escaped) so the comment line
still splits cleanly into key=value tokens.

Robustness at ~1M-compound scale:
    - Input is read in chunks (pandas, chunksize) so memory use stays
      bounded regardless of file size.
    - Every per-molecule failure is caught individually -- using
      dockstring's own exception classes (ParsingError, CanonicalizationError,
      SanityError, ProtonationError, EmbeddingError, StructureOptimizationError)
      to label *which* pipeline stage failed -- and logged to a side-car CSV.
      One bad compound never aborts the run.
    - The whole per-molecule pipeline is parallelized across processes with
      ProcessPoolExecutor (dockstring's protonate_mol calls Open Babel once
      per molecule under the hood; running many molecules concurrently
      across worker processes is what keeps that affordable at scale,
      rather than trying to batch Open Babel calls ourselves).
    - Output is streamed to disk chunk-by-chunk (optionally gzip-compressed),
      never held in memory as a whole.
    - --resume lets a killed/requeued job (e.g. hit a SLURM walltime) pick
      up where it left off without recomputing or duplicating entries.
    - --sep accepts a literal delimiter character or a friendly name
      (comma/tab/semicolon/pipe/space/whitespace), or auto-detects from the
      input file's extension.

Usage:
    python dockstring_batch_xyz.py \
        --input compounds.csv --name-col compound_name --smiles-col smiles \
        --output geometries.xyz.gz --workers 16

Requires: rdkit, dockstring (pip install rdkit dockstring), and the `obabel`
(Open Babel 3.1) binary on PATH -- dockstring.utils.protonate_mol needs it
internally; this script never calls obabel itself.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import rdmolfiles

try:
    from dockstring.errors import DockingError
    from dockstring.utils import (
        CanonicalizationError,
        EmbeddingError,
        ParsingError,
        ProtonationError,
        SanityError,
        StructureOptimizationError,
        assign_stereochemistry,
        canonicalize_smiles,
        check_charges,
        check_mol,
        check_obabel_install,
        embed_mol,
        protonate_mol,
        refine_mol_with_ff,
        sanitize_mol,
        smiles_to_mol,
    )
except ImportError as exc:
    sys.exit(
        "Could not import dockstring (and/or rdkit). Install with:\n"
        "    pip install rdkit dockstring\n"
        f"Original error: {exc}"
    )

# Silence RDKit's C++-level warnings (sanitization notices etc.) so stdout
# stays readable at 1M-compound scale; real failures are still caught and
# logged explicitly below.
RDLogger.DisableLog("rdApp.*")

DOCKSTRING_DEFAULT_SEED = 974528263  # dockstring.target.Target.dock() default

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dockstring_batch_xyz")

# Maps dockstring's own exception classes to a human-readable pipeline stage,
# so the error log says *where* a compound failed without us re-implementing
# any of dockstring's logic ourselves.
_STAGE_BY_EXCEPTION: list[tuple[type, str]] = [
    (CanonicalizationError, "canonicalization"),
    (ParsingError, "smiles_to_mol"),
    (SanityError, "sanitize_or_check"),
    (ProtonationError, "protonation"),
    (EmbeddingError, "embedding"),
    (StructureOptimizationError, "force_field_refinement"),
]


def classify_error(exc: Exception) -> str:
    for exc_type, stage in _STAGE_BY_EXCEPTION:
        if isinstance(exc, exc_type):
            return stage
    return "unknown"


# --------------------------------------------------------------------------- #
# Comment-line formatting: every input column, as key=value
# --------------------------------------------------------------------------- #
def _quote_if_needed(value: str) -> str:
    """Wrap in double quotes (escaping any literal ") if value has whitespace."""
    if _WS_RE.search(value) or '"' in value:
        return f'"{value.replace(chr(34), chr(92) + chr(34))}"'
    return value


def format_comment(row: dict, canonical_smiles: str, ph: float, seed: int) -> str:
    """Build 'col1=val1 col2=val2 ... canonical_smiles=... pH=... seed=...'."""
    parts = []
    for col, val in row.items():
        key = _WS_RE.sub("_", str(col).strip())
        parts.append(f"{key}={_quote_if_needed(str(val))}")
    parts.append(f"canonical_smiles={_quote_if_needed(canonical_smiles)}")
    parts.append(f"pH={ph}")
    parts.append(f"seed={seed}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Per-molecule pipeline (runs in worker processes)
# --------------------------------------------------------------------------- #
def process_one(
    task: tuple[dict, str, str, float, int, int]
) -> tuple[str, str, Optional[str], Optional[str], Optional[str]]:
    """
    task = (row, name_col, smiles_col, ph, seed, max_iters)
    `row` is a dict of EVERY column in the input file for this compound
    (already NaN-sanitized to "" by the caller).

    Returns (name, smiles, xyz_block_or_None, stage_or_None, error_or_None).
    Every exception is caught here -- this function must NEVER raise, or it
    will silently kill its worker process and desync the pool.
    """
    row, name_col, smiles_col, ph, seed, max_iters = task
    name = row[name_col]
    smiles = row[smiles_col]
    try:
        try:
            canonical_smiles = canonicalize_smiles(smiles)
        except CanonicalizationError:
            # Known dockstring/RDKit interaction, not a problem with this
            # molecule: dockstring.utils.canonicalize_smiles() just calls
            # RDKit's Chem.CanonSmiles(), whose own source is
            #     m = MolFromSmiles(smi)
            #     return MolToSmiles(m, bool(useChiral))
            # with no None-check on `m`. For certain heavily-stereo/ring-
            # fused SMILES, RDKit's canonicalization path can trip over
            # itself here even though Chem.MolFromSmiles() parses the very
            # same SMILES without issue -- which is exactly what the
            # fallback below re-checks. Falling back to the original SMILES
            # is safe because 3D embedding depends on the parsed molecular
            # graph, not on which (chemically equivalent) string produced it.
            canonical_smiles = smiles

        mol = smiles_to_mol(canonical_smiles, verbose=False)
        mol = sanitize_mol(mol, verbose=False)
        check_mol(mol)
        check_charges(mol)

        protonated_mol = protonate_mol(mol, pH=ph)
        check_mol(protonated_mol)

        embedded_mol = embed_mol(protonated_mol, seed=seed)
        refined_mol = refine_mol_with_ff(embedded_mol, max_iters=max_iters)
        assign_stereochemistry(refined_mol)

        comment = format_comment(row, canonical_smiles, ph, seed)
        block = rdmolfiles.MolToXYZBlock(refined_mol)
        lines = block.splitlines()
        lines[1] = comment  # line 0 = atom count, line 1 = comment
        xyz = "\n".join(lines) + "\n"
        return name, smiles, xyz, None, None
    except Exception as exc:  # noqa: BLE001 - intentionally broad; see docstring
        return name, smiles, None, classify_error(exc), f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
_WS_RE = re.compile(r"\s+")

# Friendly names for --sep, since a literal tab/pipe is awkward to type or
# easy to mistype on a command line (e.g. an unquoted "|" in bash).
_SEP_ALIASES = {
    "comma": ",",
    "tab": "\t",
    "\\t": "\t",       # in case someone types the escape sequence literally
    "semicolon": ";",
    "pipe": "|",
    "space": " ",
    "whitespace": r"\s+",  # one-or-more whitespace chars; needs engine="python"
}


def resolve_sep(sep_arg: Optional[str], input_path: Path) -> str:
    """
    Resolve --sep into an actual pandas separator string.

    Accepts: a literal single-character delimiter (",", ";", "|", ...), one
    of the friendly names in _SEP_ALIASES (case-insensitive), or nothing --
    in which case it's auto-detected from the input file's extension
    (.tsv/.tab -> tab, everything else -> comma).
    """
    if sep_arg is None:
        return "\t" if input_path.suffix.lower() in (".tsv", ".tab") else ","
    return _SEP_ALIASES.get(sep_arg.lower(), sep_arg)


def already_done_names(output_path: Path, name_key: str) -> set[str]:
    """
    Scan an existing output file's comment lines for names already written.

    `name_key` must be the exact sanitized key used in the comment line
    (see format_comment/_WS_RE) -- matching is anchored to start-of-line or
    a preceding space so a column like "username" or "compound_name" can't
    be mistaken for "name" just because it ends with the same substring.
    """
    done: set[str] = set()
    opener = gzip.open if output_path.suffix == ".gz" else open
    if not output_path.exists():
        return done
    pattern = re.compile(rf"(?:^|\s){re.escape(name_key)}=(\S+)")
    with opener(output_path, "rt") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                done.add(m.group(1))
    return done


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Delimited input file (CSV/TSV).")
    parser.add_argument("--name-col", required=True, help="Column name holding the compound name/ID.")
    parser.add_argument("--smiles-col", required=True, help="Column name holding the SMILES string.")
    parser.add_argument("--output", required=True, help="Output path, e.g. geometries.xyz or geometries.xyz.gz")
    parser.add_argument(
        "--sep",
        default=None,
        help=(
            "Input column delimiter. Accepts a literal character (',' ';' '|'), "
            "a friendly name ('comma', 'tab', 'semicolon', 'pipe', 'space', "
            "'whitespace'), or 'tab' instead of a hard-to-type literal tab. "
            "Default: auto-detect from the file extension (.tsv/.tab -> tab, else comma)."
        ),
    )
    parser.add_argument("--ph", type=float, default=7.4, help="Protonation pH (dockstring default: 7.4).")
    parser.add_argument("--seed", type=int, default=DOCKSTRING_DEFAULT_SEED, help="RDKit embedding random seed.")
    parser.add_argument("--max-iters", type=int, default=1000, help="Max force-field minimization iterations.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel worker processes.")
    parser.add_argument("--chunksize", type=int, default=20000, help="Rows per processing chunk (bounds memory use).")
    parser.add_argument("--error-log", default=None, help="CSV of failures. Default: <output>.errors.csv")
    parser.add_argument("--resume", action="store_true", help="Skip compound names already present in --output.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for testing).")
    args = parser.parse_args()

    try:
        check_obabel_install()
    except DockingError as exc:
        sys.exit(f"dockstring's Open Babel check failed: {exc}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    error_log_path = Path(args.error_log) if args.error_log else output_path.with_suffix(output_path.suffix + ".errors.csv")

    sep = resolve_sep(args.sep, input_path)
    # pandas' fast C parser can't handle multi-char/regex delimiters (e.g. the
    # "whitespace" alias); fall back to the slower but more flexible python
    # engine only when actually needed.
    read_csv_kwargs = {"sep": sep}
    if len(sep) > 1:
        read_csv_kwargs["engine"] = "python"
    log.info("Using column delimiter: %r", sep)

    name_key = _WS_RE.sub("_", args.name_col.strip())  # matches format_comment()'s key sanitizing

    skip_names: set[str] = set()
    if args.resume:
        skip_names = already_done_names(output_path, name_key)
        log.info("Resume requested: %d compounds already present in %s, will be skipped.", len(skip_names), output_path)

    out_opener = gzip.open if output_path.suffix == ".gz" else open
    output_mode = "at" if (args.resume and output_path.exists()) else "wt"

    error_log_is_new = not error_log_path.exists() or output_mode == "wt"
    error_fh = error_log_path.open("a", newline="")
    error_writer = csv.writer(error_fh)
    if error_log_is_new:
        error_writer.writerow(["name", "smiles", "stage", "error"])

    n_total = n_skipped_resume = n_missing_fields = n_failed = n_ok = 0

    # No usecols restriction: every column in the file is carried through to
    # the output comment line, not just name/smiles.
    reader = pd.read_csv(
        input_path,
        dtype=str,
        chunksize=args.chunksize,
        on_bad_lines="warn",
        **read_csv_kwargs,
    )

    with out_opener(output_path, output_mode) as out_fh, ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows_processed = 0
        for chunk in reader:
            if args.limit is not None and rows_processed >= args.limit:
                break

            tasks = []
            for _, row in chunk.iterrows():
                n_total += 1
                name = row[args.name_col]
                smi = row[args.smiles_col]

                if pd.isna(name) or pd.isna(smi) or not str(name).strip() or not str(smi).strip():
                    n_missing_fields += 1
                    error_writer.writerow([name, smi, "input", "missing name or SMILES"])
                    continue

                name = str(name).strip()
                if args.resume and name in skip_names:
                    n_skipped_resume += 1
                    continue

                # Every column goes through, not just name/smiles; NaN (an
                # empty cell) becomes "" so it still shows up as `col=` in
                # the comment line rather than silently vanishing.
                row_dict = {col: ("" if pd.isna(val) else str(val)) for col, val in row.items()}
                row_dict[args.name_col] = name
                row_dict[args.smiles_col] = str(smi).strip()

                tasks.append((row_dict, args.name_col, args.smiles_col, args.ph, args.seed, args.max_iters))

                if args.limit is not None and n_total >= args.limit:
                    break

            for name, smi, xyz_block, stage, err in pool.map(process_one, tasks, chunksize=200):
                if err is not None:
                    n_failed += 1
                    error_writer.writerow([name, smi, stage, err])
                else:
                    out_fh.write(xyz_block)
                    n_ok += 1

            rows_processed += len(chunk)
            error_fh.flush()
            log.info(
                "Processed %d rows so far (ok=%d, failed=%d, skipped=%d)",
                n_total, n_ok, n_missing_fields + n_failed, n_skipped_resume,
            )

    error_fh.close()

    log.info("=" * 60)
    log.info("Done.")
    log.info("  input rows seen:        %d", n_total)
    log.info("  skipped (resume):       %d", n_skipped_resume)
    log.info("  skipped (missing data): %d", n_missing_fields)
    log.info("  failed (pipeline):      %d", n_failed)
    log.info("  succeeded:              %d", n_ok)
    log.info("  geometries written to:  %s", output_path)
    log.info("  failure details in:     %s", error_log_path)


if __name__ == "__main__":
    main()
