#!/bin/bash
#
# executor_slurm.sh - Example SLURM Executor for profalign
#
# This is a reference implementation for cluster administrators.
# Modify according to your cluster's configuration.
#
# Usage:
#   ./executor_slurm.sh <input_dir> <output_dir> [work_dir]
#
# Example:
#   ./executor_slurm.sh ./my_profiles ./results ./work
#

set -e

# Parse arguments
INPUT=${1:?Usage: $0 <input_dir> <output_dir> [work_dir]}
OUTPUT=${2:?Usage: $0 <input_dir> <output_dir> [work_dir]}
WORK=${3:-${OUTPUT}/work}

# Configuration
SCRIPT_DIR=$(dirname "$0")
MAX_RETRIES=2
HHALIGN_TIME="00:10:00"
HHALIGN_MEM="4G"

echo "=============================================="
echo "profalign SLURM Executor"
echo "=============================================="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Work:   $WORK"
echo ""

# Step 1: Initialize
echo "=== Initializing ==="
python ${SCRIPT_DIR}/prof_align_cluster.py init "$INPUT" "$WORK"

# Create task results directory
mkdir -p "$WORK/task_results"

# Main loop
ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "=== Iteration $ITERATION ==="
    
    # Generate queue
    if ! python ${SCRIPT_DIR}/prof_align_cluster.py gen-queue "$WORK"; then
        echo "Convergence reached!"
        break
    fi
    
    # Count tasks (skip header line)
    NUM_TASKS=$(tail -n +2 "$WORK/current_queue.tsv" | grep -c . || echo 0)
    
    if [ "$NUM_TASKS" -eq 0 ]; then
        echo "No tasks to run"
        # Still process results (may trigger merges)
        touch "$WORK/results.tsv"
    else
        echo "Submitting $NUM_TASKS hhalign tasks..."
        
        # Clear previous results
        rm -f "$WORK/task_results"/*.tsv
        
        # Submit array job and wait for completion
        JOB_ID=$(sbatch --parsable \
            --time=$HHALIGN_TIME \
            --mem=$HHALIGN_MEM \
            --array=1-$NUM_TASKS \
            --output="$WORK/logs/hhalign_%A_%a.out" \
            --error="$WORK/logs/hhalign_%A_%a.err" \
            "$SCRIPT_DIR/run_hhalign_task.sh" "$WORK")
        
        echo "Submitted job $JOB_ID, waiting..."
        
        # Wait for job completion
        while squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; do
            sleep 10
        done
        
        # Collect results
        echo "Collecting results..."
        cat "$WORK/task_results"/*.tsv > "$WORK/results.tsv" 2>/dev/null || touch "$WORK/results.tsv"
        
        # Check for failed tasks
        COMPLETED=$(wc -l < "$WORK/results.tsv")
        if [ "$COMPLETED" -lt "$NUM_TASKS" ]; then
            echo "WARNING: Only $COMPLETED/$NUM_TASKS tasks completed"
            # Could add retry logic here
        fi
    fi
    
    # Process results
    echo "Processing results..."
    python ${SCRIPT_DIR}/prof_align_cluster.py process-results "$WORK" "$WORK/results.tsv"
    
    # Show status
    python ${SCRIPT_DIR}/prof_align_cluster.py status "$WORK"
done

# Finalize
echo ""
echo "=== Finalizing ==="
python ${SCRIPT_DIR}/prof_align_cluster.py finalize "$WORK" "$OUTPUT"

echo ""
echo "=============================================="
echo "Done! Results in: $OUTPUT"
echo "=============================================="
