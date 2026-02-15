import numpy as np
import xgboost as xgb
from dscribe.descriptors import SOAP
from typing import Tuple, List
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from ase.io import read
from ase import Atoms
import os
import time
import sys
import gc

# --- GLOBAL CONFIGURATION ---

TEST_DATA_PATH = 'test.xyz'
# Az eredmények ide mennek
OUTPUT_BASE_FOLDER = "OUT_FOLD"
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)
N_JOBS = 8 # Number of CPU cores for SOAP computation

# 1. PARAMETERS
if len(sys.argv) != 6:
    print("FATAL ERROR: Required 5 command line arguments are missing.")
    print("Usage: python mol_xgb.py <rcut> <nmax> <lmax> <sigma> <split_id>")
    sys.exit(1)

try:
    R = float(sys.argv[1])
    N = int(sys.argv[2])
    L = int(sys.argv[3])
    S = float(sys.argv[4])
    G = int(sys.argv[5])
except ValueError:
    print("FATAL ERROR: The parameters are invalid (type error).")
    sys.exit(1)

FILE_TAG = f"rcut{R}_nmax{N}_lmax{L}_sigma{S}_split{G}_mol"
print(f"--- Starting Grid Job: {FILE_TAG} ---")

# --- 2. SOAP PREPARATION AND SPECIES DEFINITION ---
#this try part is not needed if species is defined:
# species = set(['H','C','N',...])
try:
    train_molecules_temp = read('train.xyz', index=":")
    val_molecules_temp = read('val.xyz', index=":")
    test_molecules_temp = read(TEST_DATA_PATH, index=":") 
    all_molecules = train_molecules_temp + val_molecules_temp + test_molecules_temp
    species = set([atom for structure in all_molecules for atom in structure.get_chemical_symbols()])
    del all_molecules, train_molecules_temp, val_molecules_temp, test_molecules_temp
    gc.collect()
except FileNotFoundError:
    print("FATAL ERROR: The data (.xyz files) cannot be found in the specified paths.")
    sys.exit(1)

# Initialize SOAP (average="outer" and sparse=False due to Sklearn normalize)
# SOAP gives a dense matrix due to AVERAGE="OUTER", so the previous sparse code is not needed.
soap_calculator = SOAP(
    species=list(species), periodic=False, rcut=R, nmax=N, lmax=L, sigma=S,
    average='outer', dtype="float32", sparse=False # <-- dense SOAP
)

# --- 3. DATA PREPARATION FUNCTION (FOR AVERAGE SOAP) ---

def prepare_data_avg_output(molecules: List[Atoms], soap_calculator: SOAP) -> Tuple[xgb.DMatrix, List[str], List[float]]:
    """Calculates the averaged SOAP, normalizes it and creates a DMatrix from it."""
    
    # 1. Extracting energies and names
    energies = [mol.info['energy'] for mol in molecules]
    molecule_names = [mol.info.get('name', f'Mol_{i}') for i, mol in enumerate(molecules)]
    
    # 2. SOAP computation
    X_np = soap_calculator.create(molecules, n_jobs=N_JOBS)
    
    # 3. L2 Normalization
    X_normalized = normalize(X_np, norm='l2', axis=1)

    # 4. DMatrix preparation
    dmatrix = xgb.DMatrix(X_normalized, label=np.array(energies, dtype=np.float32))

    del X_np, X_normalized, energies
    gc.collect()

    return dmatrix, molecule_names

# --- 4. MAIN EXECUTION LOGIC (A Grid Task) ---

def run_single_job(g: int):
    
    # ------------------ TRAINING / VALIDATION PHASE -------------------

    # 1. Train Data Load
    t_train_load_start = time.time()
    train_molecules = read(f'train{g}.xyz', index=":")
    t_train_load_end = time.time()

    # 2. Train SOAP + DMatrix Prep
    t_train_soap_start = time.time()
    dtrain, train_mol_names = prepare_data_avg_output(train_molecules, soap_calculator)
    t_train_soap_end = time.time()
    
    del train_molecules; gc.collect() 

    # 3. Validation Data Load
    t_val_load_start = time.time()
    val_molecules = read(f'val{g}.xyz', index=":")
    t_val_load_end = time.time()
    
    # 4. Validation SOAP + DMatrix Prep
    t_val_soap_start = time.time()
    dval, val_mol_names = prepare_data_avg_output(val_molecules, soap_calculator)
    t_val_soap_end = time.time()
    
    del val_molecules; gc.collect() 

    # 5. Modell Training
    params = {
        "tree_method": "hist", "max_depth": 7, "learning_rate": 0.015, "gamma": 1,
        "min_child_weight": 3, "subsample": 0.75, "colsample_bytree": 1,
        "reg_alpha": 0, "reg_lambda": 1, "random_state": 0, "n_jobs": N_JOBS,
        "objective": "reg:squarederror", "eval_metric": "rmse"
    }
    t_model_train_start = time.time()
    model = xgb.train(params, dtrain, num_boost_round=5000, 
                      evals=[(dval, 'validation')],
                      early_stopping_rounds=20, verbose_eval=False)
    t_model_train_end = time.time()
    
    del dtrain, dval; gc.collect()


    # ------------------ TESTING PHASE -------------------

    # 6. TEST DATA LOAD 
    t_test_load_start = time.time()
    test_molecules = read(TEST_DATA_PATH, index=":") 
    t_test_load_end = time.time()
    
    # 7. TEST SOAP + DMatrix Prep
    t_test_soap_start = time.time() 
    dtest, test_mol_names = prepare_data_avg_output(test_molecules, soap_calculator)
    t_test_soap_end = time.time() 
    
    # We also extract the test energies for the MSE calculation
    test_energies = [mol.info['energy'] for mol in test_molecules]

    # 8. Test Prediction
    t_pred_start = time.time()
    predictions = model.predict(dtest, iteration_range=(0, model.best_iteration))
    t_pred_end = time.time()
    
    del dtest; gc.collect()

    # 9. Test RMSE Calculation
    t_rmse_start = time.time() 
    current_test_rmse = np.sqrt(mean_squared_error(test_energies, predictions))
    t_rmse_end = time.time() 
    
    # ------------------ SAVING PHASE -------------------

    # 10. SAVE RESULTS (Log, CSV)
    t_save_start = time.time() 

    # 1. Log files (Summary + Time)
    output_log_filename = os.path.join(OUTPUT_BASE_FOLDER, f"log_{FILE_TAG}.log")
    with open(output_log_filename, "w") as f:
        f.write(f"Parameters: R={R}, N={N}, L={L}, S={S}, Split={G}\n")
        f.write(f"Final Test RMSE: {current_test_rmse:.4f}\n")
        f.write(f"Best Iteration: {model.best_iteration}\n\n")
        
        f.write(f"--- Time Measurements (seconds) ---\n")
        f.write(f" 1. Train Data Load: {t_train_load_end - t_train_load_start:.2f}\n")
        f.write(f" 2. Train SOAP + DMatrix Prep: {t_train_soap_end - t_train_soap_start:.2f}\n")
        f.write(f" 3. Validation Data Load: {t_val_load_end - t_val_load_start:.2f}\n")
        f.write(f" 4. Validation SOAP + DMatrix Prep: {t_val_soap_end - t_val_soap_start:.2f}\n")
        f.write(f" 5. Model Training (XGBoost): {t_model_train_end - t_model_train_start:.2f}\n")
        f.write(f" 6. Test Data Load: {t_test_load_end - t_test_load_start:.2f}\n")
        f.write(f" 7. Test SOAP + DMatrix Prep: {t_test_soap_end - t_test_soap_start:.2f}\n")
        f.write(f" 8. Test Prediction: {t_pred_end - t_pred_start:.2f}\n")
        f.write(f" 9. Test RMSE Calculation: {t_rmse_end - t_rmse_start:.2f}\n")
    
    # 2. CSV File (Molecule prediction)
    output_csv_filename = os.path.join(OUTPUT_BASE_FOLDER, f"predictions_{FILE_TAG}.csv")
    with open(output_csv_filename, "w") as f:
        f.write("name;expected;predicted\n")
        for i in range(len(test_molecules)):
             mol_name = test_mol_names[i]
             pred_energy = predictions[i]
             true_energy = test_energies[i]
             f.write(f"{mol_name};{true_energy:.4f};{pred_energy:.4f}\n")

    t_save_end = time.time()
    
    # Update log with save time
    with open(output_log_filename, "a") as f:
         f.write(f"10. File Saving (CSV, Log): {t_save_end - t_save_start:.2f}\n")
    
    del test_molecules; gc.collect()
    print(f"Job completed successfully. Final Test RMSE: {current_test_rmse:.4f}")


if __name__ == "__main__":
    run_single_job(G)
