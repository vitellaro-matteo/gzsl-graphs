#!/bin/bash
# ============================================================
# Meluxina HPC — single-model benchmark job
#
# Submit one job per model so all 5 models run in parallel.
# Usage:
#   sbatch --export=MODEL=dgpn scripts/slurm/meluxina_single.sh
#   sbatch --export=MODEL=dbigcn scripts/slurm/meluxina_single.sh
#   sbatch --export=MODEL=icis scripts/slurm/meluxina_single.sh
#   sbatch --export=MODEL=baseline scripts/slurm/meluxina_single.sh
#   sbatch --export=MODEL=zerog scripts/slurm/meluxina_single.sh
#
# Or use submit_benchmark.sh to submit all 5 at once.
# ============================================================

#SBATCH --account=p201211     # <-- fill in your Meluxina project
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/slurm_%j_${MODEL}.out
#SBATCH --error=logs/slurm_%j_${MODEL}.err
#SBATCH --job-name=gzsl_${MODEL}

set -euo pipefail

# ---- Environment ------------------------------------------------
# Adjust CONDA_ENV to match the name of your conda environment.
CONDA_ENV="gzsl-graphs"
WORK_DIR="${SLURM_SUBMIT_DIR}"        # directory where sbatch is called from

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

cd "${WORK_DIR}"
mkdir -p logs results

echo "=============================="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Model    : ${MODEL}"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started  : $(date)"
echo "=============================="

# ---- Run experiments -------------------------------------------
python scripts/run_experiments.py \
    --config configs/full_benchmark.yaml \
    --models "${MODEL}" \
    --output_dir "results/" \
    --latex "results/table_${MODEL}.tex"

echo "Finished : $(date)"
