#!/bin/bash
# =============================================================================
# run_archive.slurm -- build the TC-diagnostics monthly archive on Negishi
#
# One array task per year (1980-2024 = 45 tasks). Each task computes the 12
# months of its year; the last task to complete a decade folds the per-month
# files into the per-decade archive files (lock-protected, so concurrent
# tasks are safe). Fully resumable: resubmitting skips finished months.
#
# Submit from the repo root:
#     mkdir -p logs && sbatch tools/run_archive.slurm
#
# ONE-TIME SETUP (login node, before first submit):
#     module load anaconda
#     conda activate research_env        # or: source activate research_env
#     pip install tcpyVPI tcpyPI
# =============================================================================
#SBATCH --job-name=tcdiag
#SBATCH --array=1980-2024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tcdiag_%A_%a.out
##SBATCH --account=chavas          # uncomment + set your allocation (`slist`);
                                   # omit to use your default account. Note the
                                   # standby queue caps walltime at 4 h.

set -euo pipefail

module purge
module load anaconda
source activate research_env       # envs live in ~/.conda/envs

cd "${SLURM_SUBMIT_DIR}"

YEAR="${SLURM_ARRAY_TASK_ID}"
OUT=/scratch/negishi/jocegue/tcdiag_archive

echo "Task ${SLURM_ARRAY_TASK_ID}: building year ${YEAR} -> ${OUT}"

python tools/build_archive.py monthly \
    --years "${YEAR}-${YEAR}" \
    --out "${OUT}"

# =============================================================================
# FOLLOW-UP: climatology (run ONCE after ALL array tasks finish -- a single
# job, NOT an array). Reads only local files, no downloads, so it also runs
# fine on a login node. Example single-job script:
#
#   #!/bin/bash
#   #SBATCH --job-name=tcdiag_clim
#   #SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#   #SBATCH --time=01:00:00
#   #SBATCH --output=logs/tcdiag_clim_%j.out
#   set -euo pipefail
#   module purge && module load anaconda
#   source activate research_env
#   cd "${SLURM_SUBMIT_DIR}"
#   OUT=/scratch/negishi/jocegue/tcdiag_archive
#   CLIM=/scratch/negishi/jocegue/tcdiag_clim
#   # 1) fold any leftover per-month files into the per-decade files:
#   python tools/build_archive.py monthly --years 1980-2024 --out "${OUT}" --consolidate-only
#   # 2) per-calendar-month + seasonal (JJA/SON/JJASON) mean/std:
#   python tools/build_archive.py climatology --years 1980-2024 --archive "${OUT}" --out "${CLIM}"
#   # 3) sanity checks:
#   python tools/build_archive.py verify --archive "${OUT}" --clim "${CLIM}"
# =============================================================================
