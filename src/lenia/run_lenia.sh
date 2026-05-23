#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=lenia-run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=lenia_run.log
#SBATCH --hint=nomultithread

set -euo pipefail

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd "$SCRIPT_DIR"

if type module >/dev/null 2>&1; then
    module load OpenMPI
fi

make clean
make

NP=${NP:-${SLURM_NTASKS:-4}}
ROWS=${ROWS:-128}
COLS=${COLS:-$ROWS}
STEPS=${STEPS:-100}
DT=${DT:-0.1}
KERNEL_SIZE=${KERNEL_SIZE:-26}
OUTPUT_FILE=${OUTPUT_FILE:-}
GIF_FILE=${GIF_FILE:-}
MPIRUN_ARGS=${MPIRUN_ARGS:---mca pml ob1}

read -r -a mpirun_args <<< "$MPIRUN_ARGS"

args=(
    --rows "$ROWS"
    --cols "$COLS"
    --steps "$STEPS"
    --dt "$DT"
    --kernel-size "$KERNEL_SIZE"
)

if [[ -n "$OUTPUT_FILE" ]]; then
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    args+=(--output "$OUTPUT_FILE")
fi

if [[ -n "$GIF_FILE" ]]; then
    mkdir -p "$(dirname "$GIF_FILE")"
    args+=(--gif "$GIF_FILE")
fi

mpirun "${mpirun_args[@]}" -np "$NP" ./lenia.out "${args[@]}"
