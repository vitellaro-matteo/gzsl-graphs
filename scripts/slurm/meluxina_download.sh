#!/bin/bash
# ============================================================
# Meluxina HPC — data download job (CPU node, no GPU needed)
#
# Downloads all 20 datasets to $SCRATCH or a project directory.
# Run this ONCE before submitting the benchmark jobs.
#
# Usage:
#   sbatch scripts/slurm/meluxina_download.sh
# ============================================================

#SBATCH --account=YOUR_PROJECT_ACCOUNT     # <-- fill in your Meluxina project
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm_%j_download.out
#SBATCH --error=logs/slurm_%j_download.err
#SBATCH --job-name=gzsl_download

set -euo pipefail

CONDA_ENV="gzsl-graphs"
WORK_DIR="${SLURM_SUBMIT_DIR}"

module load lang/Python/3.10.8-GCCcore-12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${WORK_DIR}"
mkdir -p logs data

echo "Downloading all datasets to ./data ..."
echo "Started: $(date)"

# Downloads all datasets (skips those already present)
python scripts/download_data.py --data_root ./data

echo "Finished: $(date)"
