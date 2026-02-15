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

# --- GLOBÁLIS KONFIGURÁCIÓ ÉS PARANCSSORI BEOLVASÁS ---

TEST_DATA_PATH = '/home/adam/compatible_invivo_vitro/data_splits/in-vitro-only.xyz'
# Az eredmények ide mennek
OUTPUT_BASE_FOLDER = "/home/adam/atoms_soap_xgb_allH_invivovitro_2023/grid_search_all_opt/aver"
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)
N_JOBS = 8 # CPU magok száma a SOAP számításhoz

# 1. PARAMÉTEREK BEOLVASÁSA
if len(sys.argv) != 6:
    print("FATAL ERROR: A szükséges 5 parancssori argumentum hiányzik.")
    print("Használat: python soap_avg_grid.py <rcut> <nmax> <lmax> <sigma> <split_id>")
    sys.exit(1)

try:
    R = float(sys.argv[1])
    N = int(sys.argv[2])
    L = int(sys.argv[3])
    S = float(sys.argv[4])
    G = int(sys.argv[5])
except ValueError:
    print("FATAL ERROR: A paraméterek érvénytelenek (típushiba).")
    sys.exit(1)

FILE_TAG = f"rcut{R}_nmax{N}_lmax{L}_sigma{S}_split{G}_avg"
print(f"--- Starting Grid Job: {FILE_TAG} ---")

# --- 2. SOAP ELŐKÉSZÍTÉS ÉS SPECIES DEFINIÁLÁSA ---

try:
    invivo = read('/home/adam/compatible_invivo_vitro/data_splits/in-vivo.xyz', index=":")
    test_molecules_temp = read(TEST_DATA_PATH, index=":") 
    all_molecules = invivo + test_molecules_temp
    species = set([atom for structure in all_molecules for atom in structure.get_chemical_symbols()])
    del invivo, all_molecules, test_molecules_temp
    gc.collect()
except FileNotFoundError:
    print("FATAL ERROR: Az adatok (.xyz fájlok) nem találhatók a megadott útvonalakon.")
    sys.exit(1)

# SOAP inicializálása (average="outer" és sparse=False a Sklearn normalize miatt)
# A SOAP az AVERAGE="OUTER" miatt dense (sűrű) mátrixot ad, ezért a korábbi sparse kód nem kell.
soap_calculator = SOAP(
    species=list(species), periodic=False, rcut=R, nmax=N, lmax=L, sigma=S,
    average='outer', dtype="float32", sparse=False # <-- dense SOAP
)

# --- 3. ADAT ELŐKÉSZÍTŐ FÜGGVÉNY (ÁTLAGOLT SOAP-hoz) ---

def prepare_data_avg_output(molecules: List[Atoms], soap_calculator: SOAP) -> Tuple[xgb.DMatrix, List[str], List[float]]:
    """Kiszámolja az átlagolt SOAP-ot, normalizálja és DMatrix-et készít belőle."""
    
    # 1. Energiák és nevek kinyerése
    energies = [mol.info['energy'] for mol in molecules]
    molecule_names = [mol.info.get('name', f'Mol_{i}') for i, mol in enumerate(molecules)]
    
    # 2. SOAP számítás
    X_np = soap_calculator.create(molecules, n_jobs=N_JOBS)
    
    # 3. L2 Normalizáció (Sklearn normalize használata)
    X_normalized = normalize(X_np, norm='l2', axis=1)

    # 4. DMatrix elkészítése
    dmatrix = xgb.DMatrix(X_normalized, label=np.array(energies, dtype=np.float32))

    del X_np, X_normalized, energies
    gc.collect()

    return dmatrix, molecule_names

# --- 4. FŐ VÉGREHAJTÓ LOGIKA (A Grid Task) ---

def run_single_job(g: int):
    
    # ------------------ TRÉNING / VALIDÁCIÓS FÁZIS -------------------

    # 1. Train Data Load
    t_train_load_start = time.time()
    train_molecules = read(f'/home/adam/compatible_invivo_vitro/data_splits/in-vivo_train{g}.xyz', index=":")
    t_train_load_end = time.time()

    # 2. Train SOAP KISZÁMÍTÁSA + DMatrix Prep
    t_train_soap_start = time.time()
    dtrain, train_mol_names = prepare_data_avg_output(train_molecules, soap_calculator)
    t_train_soap_end = time.time()
    
    del train_molecules; gc.collect() 

    # 3. Validation Data Load
    t_val_load_start = time.time()
    val_molecules = read(f'/home/adam/compatible_invivo_vitro/data_splits/in-vivo_val{g}.xyz', index=":")
    t_val_load_end = time.time()
    
    # 4. Validation SOAP KISZÁMÍTÁSA + DMatrix Prep
    t_val_soap_start = time.time()
    dval, val_mol_names = prepare_data_avg_output(val_molecules, soap_calculator)
    t_val_soap_end = time.time()
    
    del val_molecules; gc.collect() 

    # 5. Modell Betanítás (Tréning)
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


    # ------------------ TESZT FÁZIS -------------------

    # 6. TEST ADAT BETÖLTÉS 
    t_test_load_start = time.time()
    test_molecules = read(TEST_DATA_PATH, index=":") 
    t_test_load_end = time.time()
    
    # 7. TEST SOAP KISZÁMÍTÁSA + DMatrix Prep
    t_test_soap_start = time.time() 
    dtest, test_mol_names = prepare_data_avg_output(test_molecules, soap_calculator)
    t_test_soap_end = time.time() 
    
    # Kinyerjük a teszt energiákat is az MSE számításhoz
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
    
    # ------------------ MENTÉS FÁZIS -------------------

    # 10. EREDMÉNYEK MENTÉSE (Log, CSV)
    t_save_start = time.time() 

    # 1. Log fájl (Összefoglaló + Idők)
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
    
    # 2. CSV Fájl (Molekulánkénti predikció)
    output_csv_filename = os.path.join(OUTPUT_BASE_FOLDER, f"predictions_{FILE_TAG}.csv")
    with open(output_csv_filename, "w") as f:
        f.write("name;expected;predicted\n")
        for i in range(len(test_molecules)):
             mol_name = test_mol_names[i]
             pred_energy = predictions[i]
             true_energy = test_energies[i]
             f.write(f"{mol_name};{true_energy:.4f};{pred_energy:.4f}\n")

    t_save_end = time.time()
    
    # Log frissítése a mentési idővel
    with open(output_log_filename, "a") as f:
         f.write(f"10. File Saving (CSV, Log): {t_save_end - t_save_start:.2f}\n")
    
    del test_molecules; gc.collect()
    print(f"Job completed successfully. Final Test RMSE: {current_test_rmse:.4f}")


if __name__ == "__main__":
    run_single_job(G)
