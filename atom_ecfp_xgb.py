import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from rdkit.Chem import rdFingerprintGenerator
from typing import Tuple, List
from sklearn.metrics import mean_squared_error
import os
import time
import gc
import pandas as pd
import sys

# --- GLOBAL CONFIGURATION AND COMMAND-LINE SCANNING ---

OUTPUT_BASE_FOLDER = "OUT_FOLDER"
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)
LAMBDA = 0
N_JOBS = 8
FP_SIZE = 2048

# 1. READING PARAMETERS (<morgan_radius> <split_id>)
if len(sys.argv) != 3:
    print("FATAL ERROR: A szükséges 2 parancssori argumentum hiányzik.")
    print("Használat: python morgan_xgb_grid.py <morgan_radius> <split_id>")
    sys.exit(1)

try:
    RADIUS = int(sys.argv[1])
    G = int(sys.argv[2])
except ValueError:
    print("FATAL ERROR: A paraméterek érvénytelenek (típushiba).")
    sys.exit(1)

FILE_TAG = f"morgan_rad{RADIUS}_split{G}_2048fp"
print(f"--- Starting Morgan-XGBoost Job: {FILE_TAG} ---")

# --- 2. INDIVIDUAL ATOMIC MORGAN FEATURIZER CLASS ---
class MorganAtomFeaturizer:
    def __init__(self, radius: int = 2, fp_size: int = 256):
        self.radius = radius
        self.fp_size = fp_size
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    def __call__(self, mol: Chem.Mol) -> np.ndarray:
        n_atoms = mol.GetNumAtoms()
        atom_features = np.zeros((n_atoms, self.fp_size), dtype=np.float32)

        for atom_idx in range(n_atoms):
            fp = self.generator.GetCountFingerprint(mol, fromAtoms=[atom_idx])
            for idx, val in fp.GetNonzeroElements().items():
                if idx < self.fp_size:
                    atom_features[atom_idx, idx] = float(val)

        return atom_features

atom_featurizer = MorganAtomFeaturizer(radius=RADIUS, fp_size=FP_SIZE)

# --- 3. CUSTOM OBJECTIVE AND EVALUATION FUNCTION ---

def custom_energy_loss_with_penalty(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:

    y_true = dtrain.get_label()
    groups = dtrain.get_group()
    current_idx = 0
    grad = np.zeros_like(preds)
    hess = np.zeros_like(preds)
    penalty_lambda = LAMBDA

    for group_size in groups:
        mol_preds = preds[current_idx : current_idx + group_size]
        mol_y_true = y_true[current_idx]

        
        mol_pred_sum = np.sum(mol_preds)
        error = mol_pred_sum - mol_y_true
        grad_val = 2 * error 
        hess_val = 2 

        
        if penalty_lambda != 0 and group_size > 1:
            mean_preds = np.mean(mol_preds)
            grad_penalty = -2.0 * (mol_preds - mean_preds) / group_size
            combined_grad = grad_val + penalty_lambda * grad_penalty
        else:
            combined_grad = grad_val

        grad[current_idx : current_idx + group_size] = combined_grad
        hess[current_idx : current_idx + group_size] = hess_val

        current_idx += group_size

    return grad, hess

def custom_molecular_rmse(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """Custom evaluation metric for molecular RMSE."""
    y_true = dtrain.get_label()
    groups = dtrain.get_group()
    pred_totals = []
    y_true_totals = []
    current_idx = 0
    for group_size in groups:
        mol_preds = preds[current_idx : current_idx + group_size]
        mol_y_true = y_true[current_idx]
        pred_totals.append(np.sum(mol_preds))
        y_true_totals.append(mol_y_true)
        current_idx += group_size
    return 'molecular_rmse', np.sqrt(mean_squared_error(y_true_totals, pred_totals))

# --- 4. DATA PREPARATION FUNCTION BASED ON CSV/SMILES ---

def prepare_data_from_csv(csv_path: str, featurizer: MorganAtomFeaturizer):
    if not os.path.exists(csv_path):
        print(f"FATAL ERROR: A fájl nem található: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns or "vina_energy" not in df.columns:
        print(f"FATAL ERROR: The 'smiles' or 'energy' column is missing from the CSV!")
        sys.exit(1)

    X_list = []
    y_total_list = []
    group_sizes = []
    valid_molecules_info = []

    for _, row in df.iterrows():
        smi_str = str(row["smiles"]).strip()
        energy = float(row["vina_energy"])

        if pd.isna(energy):
            continue

        mol = Chem.MolFromSmiles(smi_str)
        if mol is not None:
            atom_feats = featurizer(mol)
            X_list.append(atom_feats)

            n_atoms = mol.GetNumAtoms()
            y_total_list.extend([energy] * n_atoms)
            group_sizes.append(n_atoms)

            mol_name = row.get("name", "N/A")
            valid_molecules_info.append({"smiles": smi_str, "energy": energy, "name": mol_name})

    if len(group_sizes) == 0:
        print("FATAL ERROR: Not a single valid molecule remained in the file!")
        sys.exit(1)

    X_full = np.vstack(X_list)
    y_full = np.array(y_total_list, dtype=np.float32)

    dmatrix = xgb.DMatrix(X_full, label=y_full)
    dmatrix.set_group(group_sizes)

    return dmatrix, group_sizes, valid_molecules_info

# --- 5. MAIN EXECUTION LOGIC ---

TEST_FILES = [
    'test.csv'  
]

def run_single_job(g: int):
    # --- PREPARATION OF TRAINING AND VALIDATION DATA WITH TIMING ---

    t_train_load_start = time.time()
    train_csv = f'train{g}.csv'
    t_train_load_end = time.time()

    t_train_feat_start = time.time()
    dtrain, train_groups, _ = prepare_data_from_csv(train_csv, atom_featurizer)
    t_train_feat_end = time.time()

    t_val_load_start = time.time()
    val_csv = f'val{g}.csv'
    t_val_load_end = time.time()

    t_val_feat_start = time.time()
    dval, val_groups, _ = prepare_data_from_csv(val_csv, atom_featurizer)
    t_val_feat_end = time.time()


    params = {
        "tree_method": "hist", "max_depth": 7, "learning_rate": 0.015, "gamma": 1,
        "min_child_weight": 3, "subsample": 0.75, "colsample_bytree": 1,
        "reg_alpha": 0, "reg_lambda": 1, "random_state": 0, "n_jobs": N_JOBS,
    }


    t_model_train_start = time.time()
    print("Starting model training...")
    model = xgb.train(params, dtrain, num_boost_round=5000, obj=custom_energy_loss_with_penalty,
                      evals=[(dval, 'validation')], feval=custom_molecular_rmse,
                      early_stopping_rounds=20, verbose_eval=50)
    t_model_train_end = time.time()

    del dtrain, dval; gc.collect()

    # --- LOOP OVER TEST FILE(S) ---
    for test_path in TEST_FILES:
        test_name = os.path.basename(test_path).replace(".csv", "")
        print(f"--- Running Inference on: {test_name} ---")

        t_test_load_start = time.time()
        t_test_load_end = time.time() 

        t_test_feat_start = time.time()
        dtest, test_groups, test_mol_infos = prepare_data_from_csv(test_path, atom_featurizer)
        t_test_feat_end = time.time()

        t_pred_start = time.time()
        test_preds_atomic = model.predict(dtest, iteration_range=(0, model.best_iteration))
        t_pred_end = time.time()
        del dtest; gc.collect()

        t_rmse_start = time.time()
        test_pred_totals = []
        test_true_totals = [info['energy'] for info in test_mol_infos]
        current_idx = 0
        for group_size in test_groups:
            test_pred_totals.append(np.sum(test_preds_atomic[current_idx:current_idx+group_size]))
            current_idx += group_size

        current_test_rmse = np.sqrt(mean_squared_error(test_true_totals, test_pred_totals))
        t_rmse_end = time.time()

        t_save_start = time.time()

        # 1. Log File (with timestamps and statistics)
        output_log_filename = os.path.join(OUTPUT_BASE_FOLDER, f"log_{FILE_TAG}_{test_name}.log")
        with open(output_log_filename, "w") as f:
            f.write(f"Test Set: {test_name}\n")
            f.write(f"Parameters: Morgan Radius={RADIUS}, Split={G}\n")
            f.write(f"Final Test RMSE: {current_test_rmse:.4f}\n")
            f.write(f"Best Iteration: {model.best_iteration}\n\n")
            f.write(f"--- Time Measurements (seconds) ---\n")
            f.write(f" 1. Train Data Load: {t_train_load_end - t_train_load_start:.4f}\n")
            f.write(f" 2. Train Featurization + DMatrix Prep: {t_train_feat_end - t_train_feat_start:.4f}\n")
            f.write(f" 3. Validation Data Load: {t_val_load_end - t_val_load_start:.4f}\n")
            f.write(f" 4. Validation Featurization + DMatrix Prep: {t_val_feat_end - t_val_feat_start:.4f}\n")
            f.write(f" 5. Model Training (XGBoost): {t_model_train_end - t_model_train_start:.4f}\n")
            f.write(f" 6. Test Data Load: {t_test_load_end - t_test_load_start:.4f}\n")
            f.write(f" 7. Test Featurization + DMatrix Prep: {t_test_feat_end - t_test_feat_start:.4f}\n")
            f.write(f" 8. Test Prediction: {t_pred_end - t_pred_start:.4f}\n")
            f.write(f" 9. Test RMSE Calculation: {t_rmse_end - t_rmse_start:.4f}\n")

        # 2. CSV File (Predictions per molecule)
        output_csv_filename = os.path.join(OUTPUT_BASE_FOLDER, f"predictions_{FILE_TAG}_{test_name}.csv")
        df_res = pd.DataFrame({
            "smiles": [info['smiles'] for info in test_mol_infos],
            "expected": test_true_totals,
            "predicted": test_pred_totals
        })
        df_res.to_csv(output_csv_filename, sep=";", index=False)

        # 3. Atomic Contributions TXT File
        output_atomic_filename = os.path.join(OUTPUT_BASE_FOLDER, f"atomic_preds_{FILE_TAG}_{test_name}.txt")
        with open(output_atomic_filename, "w") as f:
            f.write(f"--- Atomic Contributions for {FILE_TAG} on {test_name} ---\n")
            current_idx = 0
            for i, info in enumerate(test_mol_infos):
                group_size = test_groups[i]
                mol_preds = test_preds_atomic[current_idx:current_idx + group_size]
                f.write(f"\nMolecule: {info['name']}, Smiles: {info['smiles']}, True: {info['energy']:.4f}, Predicted Total: {np.sum(mol_preds):.4f}\n")

                mol = Chem.MolFromSmiles(info['smiles'])
                for j, atom in enumerate(mol.GetAtoms()):
                    f.write(f"  {atom.GetSymbol()}: {mol_preds[j]:.4f}\n")
                current_idx += group_size

        t_save_end = time.time()
        with open(output_log_filename, "a") as f:
            f.write(f" 10. File Saving (CSV, Log, Atomic): {t_save_end - t_save_start:.4f}\n")

        print(f"Finished {test_name}. Test RMSE: {current_test_rmse:.4f} | Total saving time: {t_save_end - t_save_start:.4f}s")

if __name__ == "__main__":
    run_single_job(G)
