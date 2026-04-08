#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot
import logging
import glob
from matplotlib.ticker import MaxNLocator

def main():
    logging.basicConfig(level=logging.INFO, filename='analyze_atoms.log', filemode='a', force=True)

    if len(sys.argv) < 3:
        print("Usage: python3 atom_error_graph.py /path/directory/ atom_count.csv")
        sys.exit(1)

    target_dir = sys.argv[1]
    atoms_csv_path = sys.argv[2]

    if not os.path.exists(atoms_csv_path):
        print(f"Error: Atomcount CSV not found: {atoms_csv_path}")
        sys.exit(1)
    df_atoms_all = pd.read_csv(atoms_csv_path, sep=';')

    csv_files = sorted(glob.glob(os.path.join(target_dir, "*.csv")))
    if not csv_files:
        print(f"Error: No CSV files in the target directory: {target_dir}")
        sys.exit(1)

    print(f"Processing: {len(csv_files)} files")
    run(csv_files, df_atoms_all)

def run(csv_files, df_atoms):
    mse_all, sigma_all = [], []
    mse_n_all, rel_pocet_n_all, mu_n_all, sigma_n_all = [], [], [], []
    all_n_atoms_values = []

    global_max_n = np.ceil(df_atoms['n_atoms'].max() / 10.0) * 10
    final_intervals = np.arange(1, int(global_max_n) + 11, 10)
    max_len = final_intervals.size - 1

    for f in csv_files:
        try:
            sep = ';' if "XG" not in os.path.basename(f) else ' '
            df_pred = pd.read_csv(f, sep=None, engine='python')

            df_pred = df_pred.iloc[:, [0, 1, 2]]
            df_pred.columns = ['name', 'expected', 'predicted']


            df_merged = pd.merge(df_pred, df_atoms, on='name', how='inner')

            if df_merged.empty:
                continue

            error = df_merged['expected'].values - df_merged['predicted'].values
            mse_val = np.square(error).mean()
            sigma_val = np.std(error, ddof=1)

            mse_all.append(mse_val)
            sigma_all.append(sigma_val)

            n_atoms = df_merged['n_atoms'].values
            _, mse_n, rel_n, mu_n, sig_n, _ = N_atoms_gaussian(final_intervals, n_atoms, error)

            mse_n_all.append(mse_n)
            rel_pocet_n_all.append(rel_n)
            mu_n_all.append(mu_n)
            sigma_n_all.append(sig_n)

        except Exception as e:
            print(f"Error in {f} processing: {e}")
            continue

    mse_final = np.array(mse_all)
    sigma_final = np.array(sigma_all)

    N_atoms_plot(final_intervals, np.array(mse_n_all), np.array(rel_pocet_n_all),
                 np.array(mu_n_all), np.array(sigma_n_all), mse_final, sigma_final, "N_atoms")

    matplotlib.pyplot.savefig("atom_error_graph.png", dpi=300)
    print("Done! Graph saved: atom_error_graph.png")

def N_atoms_gaussian(intervaly_hatarok, N_atoms, error):
    n_intervals = intervaly_hatarok.size - 1
    sigma_n = np.full(n_intervals, np.nan)
    mu_n = np.full(n_intervals, np.nan)
    mse_n = np.full(n_intervals, np.nan)
    rel_pocet_n = np.zeros(n_intervals)

    for i in range(n_intervals):
        mask = (N_atoms >= intervaly_hatarok[i]) & (N_atoms < intervaly_hatarok[i+1])
        if i == n_intervals - 1:
            mask = (N_atoms >= intervaly_hatarok[i])

        idx = np.where(mask)[0]
        if idx.size > 0:
            rel_pocet_n[i] = idx.size / N_atoms.size
            sigma_n[i] = np.std(error[idx], ddof=0)
            mu_n[i] = np.mean(error[idx])
            mse_n[i] = np.square(error[idx]).mean()

    return intervaly_hatarok, mse_n, rel_pocet_n, mu_n, sigma_n, None

def N_atoms_plot(intervaly_hatarok, mse_n, rel_pocet_n, mu_n, sigma_n, mse, sigma, variable_label):
    x_centers = intervaly_hatarok[:-1] + np.diff(intervaly_hatarok)/2

    interval_labels = [f"{int(intervaly_hatarok[i+1])-1}" for i in range(len(intervaly_hatarok)-1)]

    mse_n_avg = np.nanmean(mse_n, axis=0)
    rel_pocet_n_avg = np.nanmean(rel_pocet_n, axis=0)
    mu_n_avg = np.nanmean(mu_n, axis=0)
    sigma_n_avg = np.nanmean(sigma_n, axis=0)

    mse_n_std = np.nanstd(mse_n, axis=0, ddof=1)
    rel_pocet_n_std = np.nanstd(rel_pocet_n, axis=0, ddof=1)
    mu_n_std = np.nanstd(mu_n, axis=0, ddof=1)
    sigma_n_std = np.nanstd(sigma_n, axis=0, ddof=1)

    mse_avg = mse.mean()
    sigma_avg = sigma.mean()

    fig, (ax_error3, ax_error4) = matplotlib.pyplot.subplots(2, 1, figsize=(5, 7), sharex=True)
    fig.subplots_adjust(left=0.18, bottom=0.12, right=0.82, top=0.95, hspace=0)


    color_mse = 'tab:blue'
    ax_error3.set_ylabel("MSE [(kcal/mol)$^2$] --●--", color=color_mse, fontsize=12)
    ax_error3.errorbar(x_centers, mse_n_avg, yerr=mse_n_std, fmt='o--', color=color_mse, capsize=0,  markersize=7.5, linewidth=1.5, zorder=3)
    ax_error3.axhline(mse_avg, linestyle='--', color=color_mse, label=f"Avg MSE: {mse_avg:.2f}", linewidth=1.5)
    ax_error3.set_ylim(0, 4) # change if needed
    ax_error3.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_error3b = ax_error3.twinx()
    color_rel = 'tab:green'
    ax_error3b.set_ylabel("relative abundance ··■··", color=color_rel, fontsize=12)
    ax_error3b.errorbar(x_centers, rel_pocet_n_avg, yerr=rel_pocet_n_std, fmt='s:', color=color_rel, capsize=0,  markersize=7.5, linewidth=1.5, zorder=1)
    ax_error3b.set_ylim(0, 1)

    ax_error3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax_error4.set_xlabel(r"N$_{atoms}$", fontsize=12)
    ax_error4.set_xticks(x_centers)
    ax_error4.set_xticklabels(interval_labels)
    color_mu = 'tab:blue'
    ax_error4.set_ylabel("$\mu$ [kcal/mol] --●--", color=color_mu, fontsize=12)
    ax_error4.errorbar(x_centers, mu_n_avg, yerr=mu_n_std, fmt='o--', color=color_mu, capsize=0, markersize=7.5, linewidth=1.5, zorder=3)
    ax_error4.axhline(0.0, linestyle='--', color='tab:blue')

    ax_error4.set_ylim(-2,2)
    ax_error4.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_error4.set_yticks(ax_error4.get_yticks()[1:-1])

    ax_error4b = ax_error4.twinx()
    color_sig = 'tab:green'
    ax_error4b.set_ylabel("$\sigma$ [kcal/mol] ··■··", color=color_sig, fontsize=12)
    ax_error4b.errorbar(x_centers, sigma_n_avg, yerr=sigma_n_std, fmt='s:', color=color_sig, capsize=0, markersize=7.5, linewidth=1.5, zorder=1)
    ax_error4b.axhline(sigma_avg, linestyle=':', color=color_sig, label=f"Avg $\sigma$: {sigma_avg:.2f}")

    ax_error4b.set_ylim(0, 2)
    ax_error4b.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_error4b.set_yticks(ax_error4b.get_yticks()[:-1])
    ax_error4.set_xticks(x_centers)
    ax_error4.set_xticklabels(interval_labels)


    ax_error3.set_zorder(ax_error3b.get_zorder() + 1)
    ax_error3.patch.set_visible(False)
    ax_error4.set_zorder(ax_error4b.get_zorder() + 1)
    ax_error4.patch.set_visible(False)

    max_x = int(intervaly_hatarok[-1])
    print(max_x)
    ticks_val = [0] + list(np.arange(20, max_x, 20))

    fig.tight_layout()

    import matplotlib.ticker as ticker
    for ax in [ax_error3, ax_error4]:
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_val))
        ax.set_xlim(-1, max_x)

    ax_error4.set_xticklabels([str(int(t)) for t in ticks_val], fontsize=12)

    fig.subplots_adjust(hspace=0)

    return 0

if __name__ == "__main__":
    main()
