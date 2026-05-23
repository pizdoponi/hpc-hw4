#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=lenia-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=lenia_bench.log
#SBATCH --hint=nomultithread

set -euo pipefail

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd "$SCRIPT_DIR"

if type module >/dev/null 2>&1; then
    module load OpenMPI
fi

make clean
make

SIZE_LIST=${SIZE_LIST:-"128 512 1024 2048 4096"}
PROC_LIST=${PROC_LIST:-"1 2 4 16 32"}
REPEATS=${REPEATS:-5}
STEPS=${STEPS:-100}
DT=${DT:-0.1}
KERNEL_SIZE=${KERNEL_SIZE:-26}
RESULTS_DIR=${RESULTS_DIR:-benchmark_results}
EXTRA_ARGS=${EXTRA_ARGS:-}
MPIRUN_ARGS=${MPIRUN_ARGS:---mca pml ob1}
AVAILABLE_TASKS=${SLURM_NTASKS:-32}

mkdir -p "$RESULTS_DIR"
RAW_CSV="$RESULTS_DIR/lenia_raw.csv"
AVG_CSV="$RESULTS_DIR/lenia_avg.csv"

read -r -a extra_args <<< "$EXTRA_ARGS"
read -r -a mpirun_args <<< "$MPIRUN_ARGS"

printf "rows,cols,steps,procs,run,time_seconds\n" > "$RAW_CSV"

for rows in $SIZE_LIST; do
    for procs in $PROC_LIST; do
        if (( procs > AVAILABLE_TASKS )); then
            echo "Skipping rows=${rows}, procs=${procs}: not enough allocated MPI tasks."
            continue
        fi

        for run in $(seq 1 "$REPEATS"); do
            echo "Running benchmark: rows=${rows} procs=${procs} run=${run}/${REPEATS}"

            output=$(mpirun "${mpirun_args[@]}" -np "$procs" ./lenia.out \
                --rows "$rows" \
                --cols "$rows" \
                --steps "$STEPS" \
                --dt "$DT" \
                --kernel-size "$KERNEL_SIZE" \
                "${extra_args[@]}")

            echo "$output"

            runtime=$(printf '%s\n' "$output" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^time=/) {sub(/^time=/, "", $i); print $i; exit}}')

            if [[ -z "$runtime" ]]; then
                echo "Failed to parse the runtime from the program output." >&2
                exit 1
            fi

            printf "%s,%s,%s,%s,%s,%s\n" "$rows" "$rows" "$STEPS" "$procs" "$run" "$runtime" >> "$RAW_CSV"
        done
    done
done

printf "rows,cols,steps,procs,avg_time_seconds,speedup\n" > "$AVG_CSV"

for rows in $SIZE_LIST; do
    base_avg=$(awk -F, -v rows="$rows" '$1 == rows && $4 == 1 {sum += $6; count += 1} END {if (count > 0) printf "%.9f", sum / count}' "$RAW_CSV")

    if [[ -z "$base_avg" ]]; then
        continue
    fi

    for procs in $PROC_LIST; do
        avg=$(awk -F, -v rows="$rows" -v procs="$procs" '$1 == rows && $4 == procs {sum += $6; count += 1} END {if (count > 0) printf "%.9f", sum / count}' "$RAW_CSV")

        if [[ -z "$avg" ]]; then
            continue
        fi

        speedup=$(awk -v base="$base_avg" -v avg="$avg" 'BEGIN {printf "%.9f", base / avg}')
        printf "%s,%s,%s,%s,%s,%s\n" "$rows" "$rows" "$STEPS" "$procs" "$avg" "$speedup" >> "$AVG_CSV"
    done
done

echo "Raw results written to $RAW_CSV"
echo "Averaged results and speed-ups written to $AVG_CSV"
