import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from typing import Tuple, List
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import pandas as pd
import os
import time
import sys
import gc

# --- GLOBAL CONFIGURATION AND COMMAND-LINE PARSING ---

DATA_DIR = "/path/to/csv"
OUTPUT_BASE_FOLDER = "OUT_FOLD"
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)
N_JOBS = 8
FP_SIZE = 2048  

TEST_FILES = [
    os.path.join(DATA_DIR, "test.csv"),
]

# 1. PARSING PARAMETERS (<morgan_radius> <split_id>)
if len(sys.argv) != 3:
    print("FATAL ERROR: Usage: python morgan_mol_xgb_grid.py <morgan_radius> <split_id>")
    sys.exit(1)

try:
    RADIUS = int(sys.argv[1])
    G = int(sys.argv[2])
except ValueError:
    print("FATAL ERROR: Parameters are invalid (type error).")
    sys.exit(1)

FILE_TAG = f"morgan_rad{RADIUS}_split{G}_mol_2048fp"

# --- 2. MORGAN GENERATOR ---
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)

# --- 3. DATA PREPARATION FUNCTION BASED ON CSV/SMILES ---

def prepare_data_from_csv(csv_path: str) -> Tuple[xgb.DMatrix, List[str], List[str], List[float]]:
    if not os.path.exists(csv_path):
        print(f"FATAL ERROR: A fájl nem található: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns or "energy" not in df.columns:
        print(f"FATAL ERROR: The 'smiles' or 'vina_energy' column is missing from the CSV!")
        sys.exit(1)

    X_list = []
    energies = []
    molecule_names = []
    valid_smiles = []
    skipped_count = 0

    for idx, row in df.iterrows():
        smi_str = str(row["smiles"]).strip()
        energy = row["energy"]

        if pd.isna(energy):
            skipped_count += 1
            continue

        try:
            mol = Chem.MolFromSmiles(smi_str)
            if mol is not None:
                fp = morgan_generator.GetCountFingerprint(mol)
                arr = np.zeros(FP_SIZE, dtype=np.float32)
                for f_idx, val in fp.GetNonzeroElements().items():
                    if f_idx < FP_SIZE:
                        arr[f_idx] = float(val)

                X_list.append(arr)
                energies.append(float(energy))
                molecule_names.append(row.get("name", f"Mol_{idx}"))
                valid_smiles.append(smi_str)
            else:
                skipped_count += 1
        except Exception:
            skipped_count += 1

    if skipped_count > 0:
        print(f"WARNING: {csv_path} - {skipped_count} invalid/missing SMILES skipped!")

    if len(X_list) == 0:
        print(f"FATAL ERROR: No valid molecules remain in the file {csv_path}!")
        sys.exit(1)

    X_np = np.array(X_list, dtype=np.float32)
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)

    X_normalized = normalize(X_np, norm='l2', axis=1)

    dmatrix = xgb.DMatrix(X_normalized, label=np.array(energies, dtype=np.float32))
    del X_list, X_np, X_normalized
    gc.collect()

    return dmatrix, molecule_names, valid_smiles, energies

# --- 4. MAIN EXECUTION LOGIC ---

def run_single_job(g: int):
    # ------------------ TRAINING AND VALIDATION PHASE -------------------
    train_csv = os.path.join(DATA_DIR, f"train{g}.csv")
    val_csv = os.path.join(DATA_DIR, f"val{g}.csv")

    t_train_load_start = time.time()
    t_train_load_end = time.time() 

    t_train_feat_start = time.time()
    dtrain, _, _, _ = prepare_data_from_csv(train_csv)
    t_train_feat_end = time.time()

    t_val_load_start = time.time()
    t_val_load_end = time.time()

    t_val_feat_start = time.time()
    dval, _, _, _ = prepare_data_from_csv(val_csv)
    t_val_feat_end = time.time()

    params = {
        "tree_method": "hist", "max_depth": 7, "learning_rate": 0.015, "gamma": 1,
        "min_child_weight": 3, "subsample": 0.75, "colsample_bytree": 1,
        "reg_alpha": 0, "reg_lambda": 1, "random_state": 0, "n_jobs": N_JOBS,
        "objective": "reg:squarederror", "eval_metric": "rmse"
    }

    t_model_train_start = time.time()
    print("Modell tanításának indítása...")
    model = xgb.train(params, dtrain, num_boost_round=5000,
                      evals=[(dval, 'validation')],
                      early_stopping_rounds=20, verbose_eval=50)
    t_model_train_end = time.time()
    del dtrain, dval; gc.collect()

    # ------------------ ITERATING OVER TEST FILE(S) -------------------

    for test_path in TEST_FILES:
        if not os.path.exists(test_path):
            print(f"Skipping missing test file: {test_path}")
            continue

        test_id = os.path.basename(test_path).replace(".csv", "")
        CURRENT_FILE_TAG = f"morgan_rad{RADIUS}_split{G}_mol_{test_id}_2048fp"

        print(f"--- Running Inference on: {test_id} ---")

        t_test_load_start = time.time()
        t_test_load_end = time.time()

        t_test_feat_start = time.time()
        dtest, test_mol_names, test_smiles, test_true_energies = prepare_data_from_csv(test_path)
        t_test_feat_end = time.time()

        t_pred_start = time.time()
        predictions = model.predict(dtest, iteration_range=(0, model.best_iteration))
        t_pred_end = time.time()
        del dtest; gc.collect()

        t_rmse_start = time.time()
        current_test_rmse = np.sqrt(mean_squared_error(test_true_energies, predictions))
        t_rmse_end = time.time()

        # ------------------ SAVING PHASE -------------------
        t_save_start = time.time()

        # 1. Saving log file
        output_log_filename = os.path.join(OUTPUT_BASE_FOLDER, f"log_{CURRENT_FILE_TAG}.log")
        with open(output_log_filename, "w") as f:
            f.write(f"Test Set ID: {test_id}\n")
            f.write(f"Full Tag: {CURRENT_FILE_TAG}\n")
            f.write(f"Parameters: Radius={RADIUS}, Split={G}, FP_Size={FP_SIZE}\n")
            f.write(f"Final Test RMSE: {current_test_rmse:.4f}\n")
            f.write(f"Best Iteration: {model.best_iteration}\n\n")

            f.write(f"--- Time Measurements (seconds) ---\n")
            f.write(f" 1. Train Data Load: {t_train_load_end - t_train_load_start:.2f}\n")
            f.write(f" 2. Train Morgan + DMatrix Prep: {t_train_feat_end - t_train_feat_start:.2f}\n")
            f.write(f" 3. Validation Data Load: {t_val_load_end - t_val_load_start:.2f}\n")
            f.write(f" 4. Validation Morgan + DMatrix Prep: {t_val_feat_end - t_val_feat_start:.2f}\n")
            f.write(f" 5. Model Training (XGBoost): {t_model_train_end - t_model_train_start:.2f}\n")
            f.write(f" 6. Test Data Load: {t_test_load_end - t_test_load_start:.2f}\n")
            f.write(f" 7. Test Morgan + DMatrix Prep: {t_test_feat_end - t_test_feat_start:.2f}\n")
            f.write(f" 8. Test Prediction: {t_pred_end - t_pred_start:.2f}\n")
            f.write(f" 9. Test RMSE Calculation: {t_rmse_end - t_rmse_start:.2f}\n")

        # 2. Saving CSV file (predictions per molecule + SMILES)
        output_csv_filename = os.path.join(OUTPUT_BASE_FOLDER, f"predictions_{CURRENT_FILE_TAG}.csv")
        with open(output_csv_filename, "w") as f:
            f.write("smiles;expected;predicted\n")
            for i in range(len(test_true_energies)):
                f.write(f"{test_smiles[i]};{test_true_energies[i]:.4f};{predictions[i]:.4f}\n")

        t_save_end = time.time()

        with open(output_log_filename, "a") as f:
            f.write(f"10. File Saving (CSV, Log): {t_save_end - t_save_start:.2f}\n")

        print(f"Completed: {test_id} | RMSE: {current_test_rmse:.4f}")

if __name__ == "__main__":
    run_single_job(G)
