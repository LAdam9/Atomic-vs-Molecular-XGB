# Atomic vs. Molecular eXtreme Gradient Boosting - SOAP Hyperparameter optimization

This project is used to predict the energy of molecular systems using SOAP (Smooth Overlap of Atomic Positions) descriptors and XGBoost (Gradient Boosting). The code supports atomic-level energy resolution and SOAP parameter optimization on clusters (Grid Search).

## Installation and Usage
The scripts in this project are configured to run using a `singularity` container. However, you can adapt the execution to your specific environment:
* Using a different container: Replace the container path in the Slurm (`.sh`) scripts with your own that includes the required libraries.
* Using a local environment: If you prefer not to use a container, ensure you have the following Python packages installed:
```
pip install numpy xgboost dscribe ase scikit-learn scipy
```
   In this case, simply replace the `singularity exec ...` python command in the Slurm scripts with a direct `python` call. If you do not have acces to any project account, delete the `#SBATCH --account=p1156-25-1` line.

#### Local execution: 
```
python atomic_xgb.py <rcut> <nmax> <lmax> <sigma> <split_id>
```
#### HPC / Slurm Run
Scripts automatically handle the calculation of parameter combinations based on `SLURM_ARRAY_TASK_ID`:
```
sbatch atomic_xgb.sh
```

## Scripts

* `atomic_xgb.py` Atomic based model. The model examines the local environment of each atom. During training, a custom loss function (`custom_energy_loss_with_penalty`) is used:
    * The model predicts atomic energies.
    * The error is the difference between the sum of the predicted atomic energies and the true molecular energy.
    * It is also possible to add a diversity penalty.
  
* `mol_xgb.py` Molecular based model. Uses averaged SOAP vectors (`average='outer'`).
* `atomic_xgb.sh` Slurm script to run grid search of the atomic model.
* `mol_xgb.sh` Slurm script to run grid search of the averaged molecular model.

## Input
The scripts expect molecular geometries in `.xyz` format (`train{split}.xyz`, `val{split}.xyz`, `test{split}.xyz` or just one `test.xyz` file (`split=1,2,3,...`) with keys `name` and `energy`  in the comment line.

## Output

After each run, the following files are generated:
* `log_*.log`: Detailed time measurements and the final RMSE value.
* `predictions_*.csv`: Expected vs. predicted energy per molecule (`name;expected;predicted`)
* `atomic_preds_*.txt`: (Atomic model only) Energy breakdown by atomic type per molecule.

