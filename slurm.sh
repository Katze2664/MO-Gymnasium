#!/bin/bash
#
#SBATCH --job-name=minecart
#SBATCH --output=/home/ewatkins/%A_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4:00:00
#SBATCH --array=0

set -euo pipefail

SCRIPT=${1:?python script must be provided}
shift 1 # $@ holds all arguments after $1

FOZ="/fred/oz347"
ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

ml gcccore/12.2.0
ml python/3.11.2-bare
source /fred/oz347/MO-Gymnasium/.venv3_10_8/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

python -u "$SCRIPT" \
    --output_dir="$FOZ" \
    --slurm_id="$ID" \
    "$@"

cp "/home/ewatkins/$ID.txt" "$FOZ/results/$ID"
