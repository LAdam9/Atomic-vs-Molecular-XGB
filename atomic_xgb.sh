#!/bin/bash
#SBATCH -J "SOAP_XGB_GridSearch_Atomic"
#SBATCH --array=1-5%5      
#SBATCH --partition=short       
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=8     
#SBATCH -o logs/soap_xgb_grid_%A_%a.out
#SBATCH -e logs/soap_xgb_grid_%A_%a.err
#SBATCH --account=p1156-25-1 
#SBATCH --mem-per-cpu=8GB

# Fő mappák
OUTPUT_BASE_FOLDER="OUT_FOLD"
mkdir -p logs
mkdir -p "$OUTPUT_BASE_FOLDER"


# --- 1. PARAMETERERS ---
RCUTS=(6.0) #(2 3 4 5 6 ...)
N_L_COMBS=("2 1") #("1 0" "2 1" "3 0" "5 5" ...)
SIGMAS=(0.5) #(0.5 1 2 3 ...)
SPLITS=(1 2 3 4 5)

# Parameter-calculationk
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

# Create the expected output file name
FILE_TAG="rcut${RCUT}_nmax${NMAX}_lmax${LMAX}_sigma${SIGMA}_split${SPLIT}"
OUTPUT_LOG_FILE="$OUTPUT_BASE_FOLDER/log_${FILE_TAG}.log"

echo "----------------------------------------------------------------"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: RCUT=$RCUT, NMAX=$NMAX, LMAX=$LMAX, SIGMA=$SIGMA, SPLIT=$SPLIT"
echo "Expected Log: $OUTPUT_LOG_FILE"
echo "----------------------------------------------------------------"


# Check if the log file exists AND contains text indicating successful completion
if [ -f "$OUTPUT_LOG_FILE" ] && grep -q "Final Test RMSE:" "$OUTPUT_LOG_FILE"; then
    echo "✅ Task $SLURM_ARRAY_TASK_ID already completed successfully. Skipping."
    exit 0 # Exit with 0 (success), Slurm does not mark this as an error.
fi

echo "▶️ Task $SLURM_ARRAY_TASK_ID is pending or failed previously. Starting computation..."

# --- 4. Environment Setup and Run ---

module load singularity
singularity exec /storage-apps/software/containers/xgboost/tf_gpu_xgboost_04_23.sif python -u atomic_xgb.py $RCUT $NMAX $LMAX $SIGMA $SPLIT

echo "Job $SLURM_ARRAY_TASK_ID finished."
