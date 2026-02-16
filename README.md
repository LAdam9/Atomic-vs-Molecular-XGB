# Atomic vs. Molecular eXtreme Gradient Boosting - SOAP Hyperparameter optimization

This project is used to predict the energy of molecular systems using SOAP (Smooth Overlap of Atomic Positions) descriptors and XGBoost (Gradient Boosting). The code supports atomic-level energy resolution and SOAP parameter optimization on clusters (Grid Search).

## Installation and Usage
The recommended runtime is the `Singularity` container used in the project or the following Python packages:
```
pip install numpy xgboost dscribe ase scikit-learn scipy
```
##### Local execution:
```
python atomic_xgb.py <rcut> <nmax> <lmax> <sigma> <split_id>
```
##### HPC / Slurm Run
Scripts automatically handle the calculation of parameter combinations based on `SLURM_ARRAY_TASK_ID`:
```
sbatch atomic_xgb.sh
```

## Scripts

* `atomic_xgb.py` Atomic based model. The model examines the local environment of each atom. During training, a custom loss function (`custom_energy_loss_with_penalty`) is used:
** The model predicts atomic energies.
** The error is the difference between the sum of the predicted atomic energies and the true molecular energy.
** It is also possible to add a diversity penalty.
  
* `mol_xgb.py` Molecular based model. Uses averaged SOAP vectors (average='outer').
* `atomic_grid.sh` Slurm script to run grid search of the atomic model.
* `mol_avg_grid.sh` Slurm script to run grid search of the averaged molecular model.

#
