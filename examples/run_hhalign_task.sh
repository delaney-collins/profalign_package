#!/bin/bash
#
# run_hhalign_task.sh - Run single hhalign task in SLURM array
#
# Called by executor_slurm.sh as:
#   sbatch --array=1-N run_hhalign_task.sh <work_dir>
#
# Each array task reads one line from current_queue.tsv and runs hhalign.
# Output: TSV with raw SCORE (not distance) for self-score normalization.
#
# HHR parsing delegated to parse_hhr.py (single canonical parser).

#SBATCH --job-name=hhalign
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -e

WORK=${1:?Usage: $0 <work_dir>}
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

# Locate project root (parent of examples/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Read task from queue (add 1 to skip header)
LINE_NUM=$((TASK_ID + 1))
LINE=$(sed -n "${LINE_NUM}p" "$WORK/current_queue.tsv")

if [ -z "$LINE" ]; then
    echo "ERROR: No task found for TASK_ID=$TASK_ID"
    exit 1
fi

# Parse fields (tab-separated)
QUERY=$(echo "$LINE" | cut -f2)
TARGET=$(echo "$LINE" | cut -f3)
QUERY_PATH=$(echo "$LINE" | cut -f4)
TARGET_PATH=$(echo "$LINE" | cut -f5)

echo "Task $TASK_ID: $QUERY vs $TARGET"

# Create output directories
mkdir -p "$WORK/hhr" "$WORK/task_results"

# Run hhalign
OUTPUT_HHR="$WORK/hhr/${QUERY}_vs_${TARGET}.hhr"
hhalign -i "$QUERY_PATH" -t "$TARGET_PATH" -o "$OUTPUT_HHR" -M first -id 100 -diff inf -glob -v 0

# Parse result using canonical parser
python "$PROJECT_DIR/parse_hhr.py" "$OUTPUT_HHR" \
    --query "$QUERY" --target "$TARGET" \
    >> "$WORK/task_results/${TASK_ID}.tsv"

echo "Done: $QUERY vs $TARGET"
