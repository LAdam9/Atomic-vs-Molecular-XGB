#!/bin/bash
#SBATCH -J "aver"
#SBATCH --array=1-600%50      
#SBATCH --partition=short       
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=8     
#SBATCH -o logs_avg/soap_xgb_avg_%A_%a.out
#SBATCH -e logs_avg/soap_xgb_avg_%A_%a.err
#SBATCH --account=p1156-25-1 
#SBATCH --mem-per-cpu=16GB

# Fő mappák
OUTPUT_BASE_FOLDER="OUT_FOLD"
mkdir -p logs_avg
mkdir -p "$OUTPUT_BASE_FOLDER"



# --- 1. PARAMETERS (Optimization)---
RCUTS=(2.0 6.0 10.0 20.0)
N_L_COMBS=("1 0" "1 1" "2 1" "2 2" "3 3" "4 4" "5 5" "8 7" "9 8" "12 10")
SIGMAS=(0.5 1.0 2.0)
SPLITS=(1 2 3 4 5)


NUM_RCUTS=${#RCUTS[@]}
NUM_NL=${#N_L_COMBS[@]}
NUM_SIGMAS=${#SIGMAS[@]}
NUM_SPLITS=${#SPLITS[@]}
TOTAL_TASKS=$(( NUM_RCUTS * NUM_NL * NUM_SIGMAS * NUM_SPLITS ))

if [ $SLURM_ARRAY_TASK_ID -gt $TOTAL_TASKS ]; then
    echo "Task ID out of range. Expected 1 to $TOTAL_TASKS."
    exit 1
fi

TASK_ID=$(( SLURM_ARRAY_TASK_ID - 1 ))

# --- 2. INDEX CALCULATION ---
G_INDEX=$(( TASK_ID % NUM_SPLITS )); SPLIT=${SPLITS[G_INDEX]}; TASK_ID=$(( TASK_ID / NUM_SPLITS ))
S_INDEX=$(( TASK_ID % NUM_SIGMAS )); SIGMA=${SIGMAS[S_INDEX]}; TASK_ID=$(( TASK_ID / NUM_SIGMAS ))
NL_INDEX=$(( TASK_ID % NUM_NL )); NL_PAIR=${N_L_COMBS[NL_INDEX]}; NMAX=$(echo $NL_PAIR | cut -d' ' -f1); LMAX=$(echo $NL_PAIR | cut -d' ' -f2); TASK_ID=$(( TASK_ID / NUM_NL ))
R_INDEX=$(( TASK_ID % NUM_RCUTS )); RCUT=${RCUTS[R_INDEX]}


# --- 3. CHECKPOINT ---

FILE_TAG="rcut${RCUT}_nmax${NMAX}_lmax${LMAX}_sigma${SIGMA}_split${SPLIT}_mol"
OUTPUT_LOG_FILE="$OUTPUT_BASE_FOLDER/log_${FILE_TAG}.log"

echo "----------------------------------------------------------------"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: RCUT=$RCUT, NMAX=$NMAX, LMAX=$LMAX, SIGMA=$SIGMA, SPLIT=$SPLIT"
echo "Expected Log: $OUTPUT_LOG_FILE"
echo "----------------------------------------------------------------"


if [ -f "$OUTPUT_LOG_FILE" ] && grep -q "Final Test RMSE:" "$OUTPUT_LOG_FILE"; then
    echo "✅ Task $SLURM_ARRAY_TASK_ID already completed successfully. Skipping."
    exit 0 

echo "▶️ Task $SLURM_ARRAY_TASK_ID is pending or failed previously. Starting computation..."

# --- 4. Environment Setup and Run ---

module load singularity
singularity exec /storage-apps/software/containers/xgboost/tf_gpu_xgboost_04_23.sif python -u mol_xgb.py $RCUT $NMAX $LMAX $SIGMA $SPLIT

echo "Job $SLURM_ARRAY_TASK_ID finished."
