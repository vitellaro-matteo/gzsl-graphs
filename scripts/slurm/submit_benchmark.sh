#!/bin/bash
# ============================================================
# Submit the full 20-dataset benchmark to Meluxina HPC.
#
# Submits 5 separate SLURM jobs (one per model) so they all
# run in parallel on different GPU nodes.
#
# Usage (from the repo root):
#   bash scripts/slurm/submit_benchmark.sh
#
# Prerequisites:
#   1. Edit meluxina_single.sh: set --account to your project
#   2. Edit CONDA_ENV in meluxina_single.sh to match your env
#   3. Data must already be downloaded (see Step 4 in README)
# ============================================================

set -euo pipefail

SCRIPT="scripts/slurm/meluxina_single.sh"
mkdir -p logs

for MODEL in dgpn dbigcn icis baseline zerog; do
    JOB_ID=$(sbatch --export=MODEL="${MODEL}" "${SCRIPT}" | awk '{print $NF}')
    echo "Submitted ${MODEL} -> job ${JOB_ID}"
done

echo ""
echo "All 5 jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm_<JOB_ID>_<MODEL>.out"
echo ""
echo "After all jobs finish, aggregate results with:"
echo "  python scripts/run_experiments.py --aggregate_only --output_dir results/ --latex results/final_table.tex"
