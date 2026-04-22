import numpy as np
import xgboost as xgb
from dscribe.descriptors import SOAP
from typing import Tuple, List
from sklearn.metrics import mean_squared_error
from scipy.sparse import vstack as sparse_vstack, csr_matrix
from scipy.sparse.linalg import norm as sparse_norm 
import os
import time
import gc 
from ase.io import read
from ase import Atoms
import sys 

# --- GLOBAl CONFIGURATION ---

TEST_DATA_PATH = 'test.xyz'
OUTPUT_BASE_FOLDER = "OUT_FOLD"
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)
LAMBDA = 0 
N_JOBS = 8 

# 1. PARAMETERS
if len(sys.argv) != 6:
    print("FATAL ERROR: The required 5 command line arguments are missing.")
    print("Usage: python atomic_xgb.py <rcut> <nmax> <lmax> <sigma> <split_id>")
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

FILE_TAG = f"rcut{R}_nmax{N}_lmax{L}_sigma{S}_split{G}_atomic"
print(f"--- Starting Grid Job: {FILE_TAG} ---")

# --- 2. SOAP PREPARATION ---
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

soap_calculator = SOAP(
    species=list(species), periodic=False, rcut=R, nmax=N, lmax=L, sigma=S,
    average='off', dtype="float32", sparse=True
)

# --- 3. CUSTOM OBJECTIVE AND EVALUATION FUNCTIONS ---
def custom_energy_loss_with_penalty(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom Loss function with molecular energy and variance penalty.
    """
    y_true = dtrain.get_label()
    groups = dtrain.get_group()
    current_idx = 0
    grad = np.zeros_like(preds)
    hess = np.zeros_like(preds)

    try:
        penalty_lambda = 0
    except NameError:
        penalty_lambda = 0.0

    for group_size in groups:
        mol_preds = preds[current_idx : current_idx + group_size]
        mol_y_true = y_true[current_idx]
        N = group_size

        # Prediction Average
        avg_pred = np.sum(mol_preds) / N
        error = avg_pred - mol_y_true
        grad_val = 2 * error * (1 / N)
        hess_val = 2 / (N * N)

        # 3. Diversity Penalty (not neccessary)
        if penalty_lambda != 0 and N > 1:
            mean_preds = np.mean(mol_preds)
            grad_penalty = -2.0 * (mol_preds - mean_preds) / N
            combined_grad = grad_val + penalty_lambda * grad_penalty
        else:
            combined_grad = grad_val

        grad[current_idx : current_idx + N] = combined_grad
        hess[current_idx : current_idx + N] = hess_val

        current_idx += N

    return grad, hess



def custom_energy_loss_with_penalty(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    """Custom Loss function with molecular energy and variance penalty."""
    y_true = dtrain.get_label()
    groups = dtrain.get_group()
    current_idx = 0
    grad = np.zeros_like(preds)
    hess = np.zeros_like(preds)
    penalty_lambda = LAMBDA

    for group_size in groups:
        mol_preds = preds[current_idx : current_idx + group_size]
        mol_y_true = y_true[current_idx]

        # Energy summation error (MSE)
        mol_pred_sum = np.sum(mol_preds)
        error = mol_pred_sum - mol_y_true
        grad_val = 2 * error
        hess_val = 2

        # Diversity Penalty (not neccessary)
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

        predicted_molecular_value = np.sum(mol_preds) / group_size

        pred_totals.append(predicted_molecular_value)
        y_true_totals.append(mol_y_true)
        current_idx += group_size
    return 'molecular_rmse', np.sqrt(mean_squared_error(y_true_totals, pred_totals))

# --- 4. DATA PREPARATION FUNCTION ---
def prepare_data_ooc(molecules: List[Atoms], soap_calculator: SOAP):
    """  
    Calculates SOAP PER MOLECULE and builds Sparse DMatrix.  
    """
    X_list_sparse = []
    y_total_list = []
    group_sizes = []


    for mol_idx, mol in enumerate(molecules):
        # 1. Generation
        mol_soap_sparse = soap_calculator.create(mol, n_jobs=N_JOBS)
        if not isinstance(mol_soap_sparse, csr_matrix):
            mol_soap_sparse = mol_soap_sparse.tocsr()

        # 2. Normalization
        row_norms = sparse_norm(mol_soap_sparse, axis=1)
        row_norms[row_norms == 0] = 1.0
        reciprocal_norms = 1.0 / row_norms
        normalizer_matrix = csr_matrix((reciprocal_norms,
                                         (np.arange(mol_soap_sparse.shape[0]),
                                          np.arange(mol_soap_sparse.shape[0]))),
                                         shape=(mol_soap_sparse.shape[0], mol_soap_sparse.shape[0]))
        mol_soap_normalized = normalizer_matrix @ mol_soap_sparse

        # 3. Add
        X_list_sparse.append(mol_soap_normalized)

        # 4. Y data and group sizes
        n_atoms = len(mol)
        y_total_list.extend([mol.info['energy']] * n_atoms)
        group_sizes.append(n_atoms)

        # 5. INSTANT RELEASE
        del mol_soap_sparse, mol_soap_normalized

    # Vertically stacking all sparse matrices
    X_full_sparse = sparse_vstack(X_list_sparse, format='csr')
    y_full = np.array(y_total_list, dtype=np.float32)

    dmatrix = xgb.DMatrix(X_full_sparse, label=y_full)
    dmatrix.set_group(group_sizes)

    del X_list_sparse, y_total_list, X_full_sparse, y_full
    gc.collect()

    return dmatrix, group_sizes

# --- 5. MAIN EXECUTION LOGIC (A Grid Task) ---

def run_single_job(g: int):
    
    # ------------------ TRAINING / VALIDATION PHASE -------------------

    # 1. Train Data Load
    t_train_load_start = time.time()
    train_molecules = read(f'train{g}.xyz', index=":")
    t_train_load_end = time.time()

    # 2. Train SOAP + DMatrix Prep
    t_train_soap_start = time.time()
    dtrain, train_groups = prepare_data_ooc(train_molecules, soap_calculator)
    t_train_soap_end = time.time()
    
    del train_molecules; gc.collect() 

    # 3. Validation Data Load
    t_val_load_start = time.time()
    val_molecules = read(f'val{g}.xyz', index=":")
    t_val_load_end = time.time()
    
    # 4. Validation SOAP + DMatrix Prep
    t_val_soap_start = time.time()
    dval, val_groups = prepare_data_ooc(val_molecules, soap_calculator)
    t_val_soap_end = time.time()
    
    del val_molecules; gc.collect() 

    # 5. Modell Training
    params = {
        "tree_method": "hist", "max_depth": 7, "learning_rate": 0.015, "gamma": 1,
        "min_child_weight": 3, "subsample": 0.75, "colsample_bytree": 1,  
        "reg_alpha": 0, "reg_lambda": 1, "random_state": 0, "n_jobs": N_JOBS,
    }
    t_model_train_start = time.time()
    model = xgb.train(params, dtrain, num_boost_round=5000, obj=custom_energy_loss_with_penalty,
                      evals=[(dval, 'validation')], feval=custom_molecular_rmse, 
                      early_stopping_rounds=20, verbose_eval=1)
    t_model_train_end = time.time()
    
    del dtrain, dval; gc.collect()

    # ------------------ TESTING PHASE -------------------

    # 6. TEST DATA LOAD
    t_test_load_start = time.time()
    test_molecules_full = read(TEST_DATA_PATH, index=":") # or if more testing files test_molecules_full = read(f'test{g}.xyz', index=":")
    t_test_load_end = time.time()

    # 7. TEST SOAP + DMatrix Prep
    t_test_soap_start = time.time() 
    dtest, test_groups = prepare_data_ooc(test_molecules_full, soap_calculator)
    t_test_soap_end = time.time() 

    # 8. Test Prediction
    t_pred_start = time.time()
    test_preds_atomic = model.predict(dtest, iteration_range=(0, model.best_iteration))
    t_pred_end = time.time()
    
    del dtest; gc.collect()

    # 9. Test RMSE Calculation
    t_rmse_start = time.time()
    test_pred_totals = []
    test_true_totals = [mol.info['energy'] for mol in test_molecules_full]
    current_idx = 0
    for group_size in test_groups:
        predicted_total = np.sum(test_preds_atomic[current_idx:current_idx+group_size]) / group_size
        test_pred_totals.append(predicted_total)
        current_idx += group_size
    current_test_rmse = np.sqrt(mean_squared_error(test_true_totals, test_pred_totals))
    t_rmse_end = time.time()
  
    # ------------------ SAVING PHASE -------------------

    # 10. RESULTS SAVE (Log, CSV, Atomic)
    t_save_start = time.time() 

    # 1. Log files (Summary + Time)
    output_log_filename = os.path.join(OUTPUT_BASE_FOLDER, f"log_{FILE_TAG}.log")
    with open(output_log_filename, "w") as f:
        f.write(f"Parameters: R={R}, N={N}, L={L}, S={S}, Split={G}\n")
        f.write(f"Final Test RMSE: {current_test_rmse:.4f}\n")
        f.write(f"Best Iteration: {model.best_iteration}\n\n")
        
        f.write(f"--- Time Measurements (seconds) ---\n")
        f.write(f" 1. Train Data Load: {t_train_load_end - t_train_load_start:.4f}\n")
        f.write(f" 2. Train SOAP + DMatrix Prep: {t_train_soap_end - t_train_soap_start:.4f}\n")
        f.write(f" 3. Validation Data Load: {t_val_load_end - t_val_load_start:.4f}\n")
        f.write(f" 4. Validation SOAP + DMatrix Prep: {t_val_soap_end - t_val_soap_start:.4f}\n")
        f.write(f" 5. Model Training (XGBoost): {t_model_train_end - t_model_train_start:.4f}\n")
        f.write(f" 6. Test Data Load: {t_test_load_end - t_test_load_start:.4f}\n")
        f.write(f" 7. Test SOAP + DMatrix Prep: {t_test_soap_end - t_test_soap_start:.4f}\n")
        f.write(f" 8. Test Prediction: {t_pred_end - t_pred_start:.4f}\n")
        f.write(f" 9. Test RMSE Calculation: {t_rmse_end - t_rmse_start:.4f}\n")
    
    # 2. CSV Files (Molecule prediction)
    output_csv_filename = os.path.join(OUTPUT_BASE_FOLDER, f"predictions_{FILE_TAG}.csv")
    with open(output_csv_filename, "w") as f:
        f.write("name;expected;predicted\n")
        current_idx = 0
        for i, mol in enumerate(test_molecules_full):
            group_size = test_groups[i]
            predicted_total = np.sum(test_preds_atomic[current_idx:current_idx + group_size]) / group_size
            f.write(f"{mol.info.get('name', 'N/A')};{mol.info['energy']:.4f};{predicted_total:.4f}\n")
            current_idx += group_size
  
    # 3. Atomic contribution
    output_atomic_filename = os.path.join(OUTPUT_BASE_FOLDER, f"atomic_preds_{FILE_TAG}.txt")
    with open(output_atomic_filename, "w") as f:
        f.write(f"--- Atomic Contributions for {FILE_TAG} ---\n")
        current_idx = 0
        for i, mol in enumerate(test_molecules_full):
            group_size = test_groups[i]
            mol_preds = test_preds_atomic[current_idx:current_idx + group_size]
            predicted_average = np.sum(mol_preds) / group_size
            f.write(f"\nMolecule: {mol.info.get('name', 'N/A')}, True: {mol.info['energy']:.4f}, Predicted Average: {predicted_average:.4f}\n")
            for j, atom in enumerate(mol):
                f.write(f" {atom.symbol}: {mol_preds[j]:.4f}\n")
            current_idx += group_size
    t_save_end = time.time()

                                
    # Update log with save time
    with open(output_log_filename, "a") as f:
         f.write(f"10. File Saving (CSV, Log, Atomic): {t_save_end - t_save_start:.4f}\n")
    
    del test_molecules_full; gc.collect()
    print(f"Job completed successfully. Total saving time: {t_save_end - t_save_start:.4f}s")


if __name__ == "__main__":
    run_single_job(G)
