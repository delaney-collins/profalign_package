#!/usr/bin/env python3
"""
executor_local.py - Local mock Executor for testing cluster mode

This script simulates a cluster Executor by:
1. Calling prof_align_cluster.py commands
2. Running hhalign tasks locally (sequentially or in parallel)
3. Collecting results and continuing to next iteration

Usage:
    python executor_local.py <input_dir> <output_dir> [options]

Example:
    python executor_local.py ../data/msa ./results_cluster --threads 4

This serves as:
1. A testing tool for the cluster mode pipeline
2. A reference implementation for cluster administrators

All preprocessing (filter + consensus) is done by prof_align_cluster.py init.

Author: profalign team
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_version():
    """Read version from VERSION file."""
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "unknown"


@dataclass
class Task:
    """A single hhalign task."""
    task_id: str
    query: str
    target: str
    query_path: Path
    target_path: Path


@dataclass 
class TaskResult:
    """Result of a single hhalign task."""
    query: str
    target: str
    score: float  # Raw HHalign score (not distance!)
    probability: float
    aligned_cols: int
    coverage: float


def run_command(cmd: List[str], description: str = "") -> subprocess.CompletedProcess:
    """Run a command and handle errors."""
    logger.debug(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0 and result.returncode != 1:
        # returncode 1 is used for "no more work" signals
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"stderr: {result.stderr}")
    
    return result


def read_queue(queue_file: Path) -> List[Task]:
    """Read task queue from TSV file."""
    tasks = []
    
    if not queue_file.exists():
        return tasks
    
    with open(queue_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                tasks.append(Task(
                    task_id=parts[0],
                    query=parts[1],
                    target=parts[2],
                    query_path=Path(parts[3]),
                    target_path=Path(parts[4])
                ))
    
    return tasks


def run_hhalign_task(task: Task, work_dir: Path) -> Optional[TaskResult]:
    """
    Run a single hhalign task.
    
    NOTE: Input files are already prepared (filtered + consensus) by
    prof_align_cluster.py init. We just run hhalign directly.
    """
    # Import required modules
    sys.path.insert(0, str(Path(__file__).parent / "lib"))
    try:
        from hhsuite import hhalign
        from hhr_parser import parse_hhr_file
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        return None
    
    hhr_dir = work_dir / "hhr"
    hhr_dir.mkdir(exist_ok=True)
    
    try:
        # Run hhalign directly on prepared files (already have filter + consensus)
        hhr_file = hhr_dir / f"{task.query}_vs_{task.target}.hhr"
        
        try:
            hhalign(
                query=task.query_path,
                template=task.target_path,
                output=hhr_file,
                global_alignment=True
            )
            success = hhr_file.exists()
        except Exception as e:
            logger.warning(f"hhalign error: {e}")
            success = False
        
        if not success or not hhr_file.exists():
            logger.warning(f"hhalign failed for {task.query} vs {task.target}")
            return TaskResult(
                query=task.query,
                target=task.target,
                score=0.0,
                probability=0.0,
                aligned_cols=0,
                coverage=0.0
            )
        
        # Parse results
        result = parse_hhr_file(hhr_file)
        
        if not result.hits:
            return TaskResult(
                query=task.query,
                target=task.target,
                score=0.0,
                probability=0.0,
                aligned_cols=0,
                coverage=0.0
            )
        
        hit = result.hits[0]
        score = hit.score
        probability = hit.probability
        aligned_cols = hit.aligned_cols
        
        # Calculate coverage (raw, no filtering - filtering done in process-results)
        query_len = result.query_length if result.query_length > 0 else 1
        target_len = hit.template_length if hit.template_length > 0 else 1
        query_cov = aligned_cols / query_len
        target_cov = aligned_cols / target_len
        coverage = min(query_cov, target_cov)
        
        return TaskResult(
            query=task.query,
            target=task.target,
            score=score,
            probability=probability,
            aligned_cols=aligned_cols,
            coverage=coverage
        )
        
    except Exception as e:
        logger.error(f"Error processing {task.query} vs {task.target}: {e}")
        return TaskResult(
            query=task.query,
            target=task.target,
            score=0.0,
            probability=0.0,
            aligned_cols=0,
            coverage=0.0
        )


def execute_tasks(tasks: List[Task], work_dir: Path, 
                  threads: int = 1) -> List[TaskResult]:
    """Execute all tasks and return results."""
    results = []
    
    if threads == 1:
        # Sequential execution
        for i, task in enumerate(tasks, 1):
            logger.info(f"  Task {i}/{len(tasks)}: {task.query} vs {task.target}")
            result = run_hhalign_task(task, work_dir)
            if result:
                results.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {
                executor.submit(run_hhalign_task, task, work_dir): task 
                for task in tasks
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                task = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    if completed % 10 == 0 or completed == len(tasks):
                        logger.info(f"  Completed {completed}/{len(tasks)} tasks")
                except Exception as e:
                    logger.error(f"Task failed: {task.query} vs {task.target}: {e}")
    
    return results


def write_results(results: List[TaskResult], output_file: Path):
    """Write results to TSV file."""
    # Sort for deterministic output (parallel execution returns in arbitrary order)
    results_sorted = sorted(results, key=lambda r: (r.query, r.target))
    with open(output_file, 'w') as f:
        f.write("# query\ttarget\tscore\tprobability\taligned_cols\tcoverage\n")
        for r in results_sorted:
            f.write(f"{r.query}\t{r.target}\t{r.score:.1f}\t{r.probability:.2f}\t{r.aligned_cols}\t{r.coverage:.4f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Local mock Executor for testing cluster mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python executor_local.py ../data/msa ./results_cluster
  
  # Run with 4 threads
  python executor_local.py ../data/msa ./results_cluster --threads 4
  
  # Custom parameters
  python executor_local.py ../data/msa ./results_cluster \\
      --depth-threshold 1.2 \\
      --length-coverage 0.667

This script serves as both a testing tool and a reference implementation
for cluster administrators who need to write their own Executor.
"""
    )
    
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {get_version()}')
    
    parser.add_argument('input_dir', type=Path, help='Input directory with MSA files')
    parser.add_argument('output_dir', type=Path, help='Output directory')
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads (default: 1)')
    parser.add_argument('--depth-threshold', type=float, default=1.2, help='Depth threshold for tree cutting')
    parser.add_argument('--depth-weight-mode', type=int, default=2, choices=[0, 1, 2], help='Depth weighting mode')
    parser.add_argument('--merge-gap-threshold', type=float, default=0.667, help='Gap threshold for merge')
    parser.add_argument('--distance-gap-threshold', type=float, default=0.5, help='Gap threshold for distance/filtering')
    parser.add_argument('--length-coverage', type=float, default=0.667, 
                        help='Min alignment coverage (default: 0.667)')
    parser.add_argument('--max-iterations', type=int, default=10, help='Maximum iterations')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Paths
    script_dir = Path(__file__).parent
    cluster_script = script_dir / "prof_align_cluster.py"
    work_dir = args.output_dir / "work"
    
    if not cluster_script.exists():
        logger.error(f"Cluster script not found: {cluster_script}")
        return 1
    
    logger.info("=" * 60)
    logger.info(f"Local Executor for Profile Alignment (v{get_version()})")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Threads: {args.threads}")
    logger.info("")
    
    # Step 1: Initialize
    logger.info("=== Step 1: Initialize ===")
    cmd = [
        sys.executable, str(cluster_script), 'init',
        str(args.input_dir), str(work_dir),
        '--depth-threshold', str(args.depth_threshold),
        '--depth-weight-mode', str(args.depth_weight_mode),
        '--merge-gap-threshold', str(args.merge_gap_threshold),
        '--distance-gap-threshold', str(args.distance_gap_threshold),
        '--length-coverage', str(args.length_coverage),
        '--max-iterations', str(args.max_iterations),
    ]
    result = run_command(cmd, "init")
    if result.returncode != 0:
        logger.error("Init failed")
        return 1
    
    # Main loop
    iteration = 0
    while True:
        iteration += 1
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"=== Iteration {iteration} ===")
        logger.info(f"{'='*60}")
        
        # Step 2: Generate queue
        logger.info("Generating task queue...")
        cmd = [sys.executable, str(cluster_script), 'gen-queue', str(work_dir)]
        result = run_command(cmd, "gen-queue")
        
        if result.returncode != 0:
            logger.info("Pipeline converged or max iterations reached")
            break
        
        # Read queue
        queue_file = work_dir / "current_queue.tsv"
        tasks = read_queue(queue_file)
        
        logger.info(f"Tasks in queue: {len(tasks)}")
        
        # Step 3: Execute tasks
        if tasks:
            logger.info("Executing hhalign tasks...")
            results = execute_tasks(tasks, work_dir, threads=args.threads)
            
            # Write results
            results_file = work_dir / "results.tsv"
            write_results(results, results_file)
            logger.info(f"Results written: {len(results)} pairs")
        else:
            # Empty queue - still need to process for potential merges
            results_file = work_dir / "results.tsv"
            with open(results_file, 'w') as f:
                f.write("# query\ttarget\tscore\tprobability\taligned_cols\tcoverage\n")
        
        # Step 4: Process results
        logger.info("Processing results...")
        cmd = [
            sys.executable, str(cluster_script), 'process-results',
            str(work_dir), str(results_file)
        ]
        result = run_command(cmd, "process-results")
        
        # Check status
        cmd = [sys.executable, str(cluster_script), 'status', str(work_dir)]
        result = run_command(cmd, "status")
        print(result.stdout)
        
        # Check if converged
        import json
        state_file = work_dir / "state.json"
        with open(state_file) as f:
            state = json.load(f)
        
        if state['status'] == 'converged':
            logger.info("Pipeline converged!")
            break
    
    # Step 5: Finalize
    logger.info("")
    logger.info("=== Finalizing ===")
    cmd = [
        sys.executable, str(cluster_script), 'finalize',
        str(work_dir), str(args.output_dir)
    ]
    result = run_command(cmd, "finalize")
    
    # Print final summary
    summary_file = args.output_dir / "summary.txt"
    if summary_file.exists():
        print("\n" + "=" * 60)
        print(summary_file.read_text())
        print("=" * 60)
    
    logger.info(f"\nDone! Results in: {args.output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
