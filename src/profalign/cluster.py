#!/usr/bin/env python3
"""
prof_align_cluster.py - Cluster mode CLI for profile alignment pipeline

Provides stateless commands for external Executor to manage the pipeline:
  - init: Initialize work directory
  - gen-queue: Generate task queue for current iteration
  - process-results: Process hhalign results and perform merges
  - status: Show current pipeline status
  - finalize: Generate final output

Usage:
    python prof_align_cluster.py init <input_dir> <work_dir> [options]
    python prof_align_cluster.py gen-queue <work_dir> [options]
    python prof_align_cluster.py process-results <work_dir> <results_file>
    python prof_align_cluster.py status <work_dir>
    python prof_align_cluster.py finalize <work_dir> <output_dir>

This script is designed to be called by an external Executor (e.g., SLURM wrapper)
that handles the actual job submission and collection.

Author: profalign team
"""

import argparse
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


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


# =============================================================================
# Profile Preparation
# =============================================================================

def prepare_profile_for_hhalign(input_path: Path, output_path: Path,
                                 profile_name: str,
                                 gap_threshold: float = 0.5) -> int:
    """
    Prepare a profile for hhalign by filtering gappy columns and adding consensus.
    
    This matches executor_local.py behavior:
    1. Read raw MSA
    2. Filter columns with gap_fraction > gap_threshold
    3. Add consensus sequence as first sequence
    4. Save to output_path
    
    Args:
        input_path: Path to raw MSA file
        output_path: Path to save prepared profile
        profile_name: Name for consensus sequence (e.g., "C25_Cas10_1")
        gap_threshold: Max gap fraction for column filtering (default: 0.5)
        
    Returns:
        Number of columns after filtering
    """
    from .io.formats import read_fasta, write_fasta
    from .alignment.merge import filter_alignment_columns, extract_filtered_sequences, add_consensus_to_alignment
    
    # Read raw alignment
    alignment = read_fasta(input_path)
    original_cols = alignment.get_length()
    
    # Filter gappy columns
    if gap_threshold < 1.0:
        filtered_info = filter_alignment_columns(alignment, gap_threshold=gap_threshold)
        filtered_ali = extract_filtered_sequences(filtered_info)
        filtered_cols = filtered_ali.get_length()
    else:
        filtered_ali = alignment
        filtered_cols = original_cols
    
    # Add consensus sequence
    consensus_name = f"{profile_name}@_cons"
    ali_with_cons = add_consensus_to_alignment(filtered_ali, consensus_name)
    
    # Save
    write_fasta(ali_with_cons, output_path)
    
    return filtered_cols

def cmd_run_task(args):
    """Run one hhalign task from current_queue.tsv."""
    import csv
    import subprocess

    work_dir = Path(args.work_dir)
    task_id = str(args.task_id).zfill(6)

    queue_file = work_dir / "current_queue.tsv"
    results_dir = work_dir / "task_results"
    hhr_dir = work_dir / "hhr"

    results_dir.mkdir(exist_ok=True)
    hhr_dir.mkdir(exist_ok=True)

    if not queue_file.exists():
        logger.error(f"Queue file not found: {queue_file}")
        return 1

    task = None
    with open(queue_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue

            row_task_id, query, target, query_path, target_path = parts[:5]

            if row_task_id == task_id or row_task_id.lstrip("0") == str(args.task_id):
                task = {
                    "task_id": row_task_id,
                    "query": query,
                    "target": target,
                    "query_path": Path(query_path),
                    "target_path": Path(target_path),
                }
                break

    if task is None:
        logger.error(f"Task ID not found in queue: {args.task_id}")
        return 1

    hhr_path = hhr_dir / f"{task['query']}__vs__{task['target']}.hhr"
    result_path = results_dir / f"{task_id}.tsv"
 
    from .pipeline.task_queue import PairwiseTask, run_single_task

    pair_task = PairwiseTask(
        query_profile=task["query_path"],
        target_profile=task["target_path"],
        query_name=task["query"],
        target_name=task["target"],
        output_hhr=hhr_path,
    )

    if not (work_dir /"state.json").exists():
        logger.error("State file not found. Run 'cluster init' first.")
        return 1

    state = PipelineState.load(work_dir / "state.json")

    result = run_single_task(
        pair_task,
        add_consensus=True,
        min_coverage=state.coverage_threshold,
        gap_threshold=state.distance_gap_threshold,
    )

    if not result.success:
        logger.error(f"Task failed: {result.error_message}")
        return 1
    
    with open(result_path, "w") as out:
        out.write(
            f"{result.query_name}\t"
            f"{result.target_name}\t"
            f"{result.score}\t"
            f"{result.probability}\t"
            f"{result.aligned_cols}\t"
            f"{result.coverage}\n"
        )

    logger.info(f"Result written: {result_path}")
    return 0


# =============================================================================
# State Management
# =============================================================================

@dataclass
class PipelineState:
    """Pipeline state persisted between calls."""
    iteration: int = 0
    active_profiles: List[str] = None
    merged_away: List[str] = None
    total_merges: int = 0
    pairs_computed: int = 0
    status: str = "initialized"  # initialized, queue_ready, processing, converged
    
    # Config (set at init, preserved across iterations)
    depth_threshold: float = 1.2
    depth_weight_mode: int = 2
    merge_gap_threshold: float = 0.667
    distance_gap_threshold: float = 0.5
    coverage_threshold: float = 0.667  # renamed CLI: --length-coverage
    max_iterations: int = 10
    
    def __post_init__(self):
        if self.active_profiles is None:
            self.active_profiles = []
        if self.merged_away is None:
            self.merged_away = []
    
    def save(self, filepath: Path):
        """Save state to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'PipelineState':
        """Load state from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# Command: init
# =============================================================================

def cmd_init(args):
    """Initialize work directory for cluster mode."""
    input_dir = Path(args.input_dir)
    work_dir = Path(args.work_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # Create work directory structure
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "profiles").mkdir(exist_ok=True)      # Prepared for hhalign
    (work_dir / "profiles_raw").mkdir(exist_ok=True)  # Original MSAs (for merge)
    (work_dir / "merged").mkdir(exist_ok=True)        # Prepared merged profiles
    (work_dir / "merged_raw").mkdir(exist_ok=True)    # Raw merged profiles (for further merge)
    (work_dir / "hhr").mkdir(exist_ok=True)
    (work_dir / "iterations").mkdir(exist_ok=True)
    
    # Import functions for profile preparation
    try:
        from .io.formats import read_fasta, write_fasta
        from .alignment.merge import filter_alignment_columns, extract_filtered_sequences, add_consensus_to_alignment
    except ImportError as e:
        logger.exception("Failed to import required profalign modules")
        logger.error("This usually indicates an installation or package structure problem. "
                     "Expected modules: profalign.io.formats and profalign.alignment.merge")
        return 1
    
    # Process input profiles
    profiles = []
    gap_threshold = args.distance_gap_threshold  # Use distance gap threshold for filtering
    
    for ext in ['*.fasta', '*.fa', '*.a3m']:
        for src in input_dir.glob(ext):
            profile_name = src.stem
            
            # Save raw copy (for merge operations)
            raw_dst = work_dir / "profiles_raw" / src.name
            shutil.copy2(src, raw_dst)
            
            # Prepare filtered + consensus version (for hhalign)
            prepared_dst = work_dir / "profiles" / src.name
            
            try:
                filtered_cols = prepare_profile_for_hhalign(
                    src, prepared_dst, profile_name,
                    gap_threshold=gap_threshold
                )
                
                # Read original to get stats
                original_ali = read_fasta(src)
                original_cols = original_ali.get_length()
                
                profiles.append(profile_name)
                logger.info(f"  Processed: {src.name} ({len(original_ali)} seqs, "
                           f"{filtered_cols}/{original_cols} cols after filter)")
                
            except Exception as e:
                logger.warning(f"  Failed to process {src.name}: {e}, copying as-is")
                shutil.copy2(src, prepared_dst)
                profiles.append(profile_name)
    
    profiles.sort()  # Deterministic order
    
    if not profiles:
        logger.error("No profile files found in input directory")
        return 1
    
    logger.info(f"Initialized with {len(profiles)} profiles")
    logger.info(f"  profiles_raw/: original MSAs (for merge)")
    logger.info(f"  profiles/: filtered + consensus (for hhalign, gap_threshold={gap_threshold})")
    
    # Initialize state
    state = PipelineState(
        iteration=0,
        active_profiles=profiles,
        merged_away=[],
        total_merges=0,
        pairs_computed=0,
        status="initialized",
        depth_threshold=args.depth_threshold,
        depth_weight_mode=args.depth_weight_mode,
        merge_gap_threshold=args.merge_gap_threshold,
        distance_gap_threshold=args.distance_gap_threshold,
        coverage_threshold=args.length_coverage,
        max_iterations=args.max_iterations,
    )
    state.save(work_dir / "state.json")
    
    # Initialize empty scores file
    distances_file = work_dir / "distances.tsv"
    with open(distances_file, 'w') as f:
        f.write("# profile_a\tprofile_b\tscore\tprobability\taligned_cols\tcoverage\n")
    
    # Initialize empty merge history
    history_file = work_dir / "merge_history.tsv"
    with open(history_file, 'w') as f:
        f.write("# merge_id\tparent1\tparent2\titeration\n")
    
    logger.info(f"Work directory ready: {work_dir}")
    return 0


# =============================================================================
# Command: gen-queue
# =============================================================================

def cmd_gen_queue(args):
    """Generate task queue for current iteration."""
    work_dir = Path(args.work_dir)
    
    state_file = work_dir / "state.json"
    if not state_file.exists():
        logger.error("State file not found. Run 'init' first.")
        return 1
    
    state = PipelineState.load(state_file)
    
    # Check for convergence
    if state.status == "converged":
        logger.info("Pipeline already converged")
        return 1
    
    if state.iteration >= state.max_iterations:
        logger.info(f"Max iterations ({state.max_iterations}) reached")
        state.status = "converged"
        state.save(state_file)
        return 1
    
    # Increment iteration
    state.iteration += 1
    logger.info(f"=== Generating queue for iteration {state.iteration} ===")
    
    # Load existing scores (directional keys!)
    distances_file = work_dir / "distances.tsv"
    computed_pairs = set()
    if distances_file.exists():
        with open(distances_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    key = (parts[0], parts[1])
                    computed_pairs.add(key)
    
    # Generate all pairs needed
    active = sorted(state.active_profiles)
    all_pairs = []
    
    # Self-comparisons first (needed for score normalization)
    for p in active:
        key = (p, p)
        if key not in computed_pairs:
            all_pairs.append((p, p))
    
    # All pairs in BOTH directions (for asymmetric normalization)
    for i, p1 in enumerate(active):
        for j, p2 in enumerate(active):
            if i != j:
                key = (p1, p2)
                if key not in computed_pairs:
                    all_pairs.append((p1, p2))
    
    logger.info(f"Active profiles: {len(active)}")
    logger.info(f"Pairs already computed: {len(computed_pairs)}")
    logger.info(f"New pairs to compute: {len(all_pairs)}")
    
    if not all_pairs:
        logger.info("No new pairs to compute, proceeding to merge step")
        state.status = "queue_ready"
        state.save(state_file)
        
        queue_file = work_dir / "current_queue.tsv"
        with open(queue_file, 'w') as f:
            f.write("# task_id\tquery\ttarget\tquery_path\ttarget_path\n")
        
        return 0
    
    # Create iteration directory
    iter_dir = work_dir / "iterations" / f"iter_{state.iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    # Write queue file
    # NOTE: Paths point to PREPARED profiles (filtered + consensus)
    queue_file = work_dir / "current_queue.tsv"
    with open(queue_file, 'w') as f:
        f.write("# task_id\tquery\ttarget\tquery_path\ttarget_path\n")
        for idx, (p1, p2) in enumerate(all_pairs, 1):
            p1_path = find_profile_path(work_dir, p1, prepared=True)
            p2_path = find_profile_path(work_dir, p2, prepared=True)
            f.write(f"{idx:06d}\t{p1}\t{p2}\t{p1_path}\t{p2_path}\n")
    
    logger.info(f"Queue written: {queue_file}")
    logger.info(f"Tasks: {len(all_pairs)}")
    
    state.status = "queue_ready"
    state.save(state_file)
    
    return 0


def find_profile_path(work_dir: Path, profile_name: str, prepared: bool = True) -> Path:
    """
    Find profile file path.
    
    Args:
        work_dir: Work directory
        profile_name: Profile name (without extension)
        prepared: If True, return prepared (filtered+consensus) version for hhalign
                  If False, return raw version for merge
    """
    if prepared:
        # For hhalign: use prepared versions
        subdirs = ["merged", "profiles"]
    else:
        # For merge: use raw versions
        subdirs = ["merged_raw", "profiles_raw"]
    
    for subdir in subdirs:
        for ext in [".fasta", ".fa", ".a3m"]:
            path = work_dir / subdir / f"{profile_name}{ext}"
            if path.exists():
                return path
    
    raise FileNotFoundError(f"Profile not found: {profile_name} (prepared={prepared})")


# =============================================================================
# Command: process-results
# =============================================================================

def cmd_process_results(args):
    """Process hhalign results and perform merges."""
    work_dir = Path(args.work_dir)
    results_file = Path(args.results_file) if args.results_file else work_dir / "results.tsv"
    
    state_file = work_dir / "state.json"
    if not state_file.exists():
        logger.error("State file not found. Run 'init' first.")
        return 1
    
    state = PipelineState.load(state_file)
    
    logger.info(f"=== Processing results for iteration {state.iteration} ===")
    
    # Append new results to distances.tsv
    distances_file = work_dir / "distances.tsv"
    new_pairs = 0
    
    if results_file.exists() and results_file.stat().st_size > 0:
        with open(results_file) as f_in, open(distances_file, 'a') as f_out:
            for line in f_in:
                if line.startswith('#'):
                    continue
                if line.strip():
                    f_out.write(line)
                    new_pairs += 1
        
        logger.info(f"Added {new_pairs} new score pairs")
        state.pairs_computed += new_pairs
    
    # Load all scores
    scores = load_scores(distances_file)
    logger.info(f"Total scores loaded: {len(scores)}")
    
    # Build distance matrix for active profiles
    active = sorted(state.active_profiles)
    if len(active) < 2:
        logger.info("Less than 2 profiles remaining — converged")
        state.status = "converged"
        state.save(state_file)
        return 0
    
    # Check for missing self-scores
    missing_self = [p for p in active if (p, p) not in scores]
    if missing_self:
        logger.warning(f"Missing self-scores for {len(missing_self)} profiles: {missing_self[:5]}...")
    
    # Import clustering functions
    from .clustering.clustering import hierarchical_clustering, get_tree_height
    from .alignment.guided_align import (
        scipy_node_to_tree_node, calc_tree_weighted_depth,
        extract_subtrees_at_depth, get_nontrivial_subtrees,
        run_guided_alignment, GuidedMergeRecord
    )
    from .alignment.merge import merge_two_msas
    from .pipeline.pipeline_utils import generate_merge_id
    from scipy.cluster.hierarchy import to_tree
    
    # Create merge wrapper that uses RAW profiles
    def merge_wrapper(path1: Path, path2: Path, output_path: Path,
                      gap_threshold: float = 0.667,
                      homogeneity_threshold: float = 0.0):
        """Wrapper for merge_two_msas using RAW profiles."""
        return merge_two_msas(
            path1, path2, output_path,
            gap_threshold=gap_threshold,
            homogeneity_threshold=homogeneity_threshold,
            global_alignment=True,
            work_dir=work_dir / "merge_tmp" / output_path.stem
        )
    
    # Build distance matrix
    matrix = build_distance_matrix(
        scores, 
        active,
        min_coverage=state.coverage_threshold,
        max_distance=12.0
    )
    
    # Build UPGMA tree
    logger.info("Building UPGMA tree...")
    linkage = hierarchical_clustering(matrix / 2.0, method='upgma')
    tree = to_tree(linkage)
    tree_height = get_tree_height(linkage)
    logger.info(f"Tree height: {tree_height:.3f}")
    
    # Convert to TreeNode and calculate weighted depth
    tree_node = scipy_node_to_tree_node(tree, active)
    calc_tree_weighted_depth(tree_node, weight_mode=state.depth_weight_mode)
    
    # Extract subtrees at depth threshold
    subtrees = extract_subtrees_at_depth(tree_node, state.depth_threshold)
    nontrivial = get_nontrivial_subtrees(subtrees)
    
    logger.info(f"Subtrees: {len(subtrees)} total, {len(nontrivial)} non-trivial")
    
    if not nontrivial:
        logger.info("All subtrees are singletons — converged")
        state.status = "converged"
        state.save(state_file)
        return 0
    
    # Perform merges
    merge_history_file = work_dir / "merge_history.tsv"
    merged_dir = work_dir / "merged"
    merged_raw_dir = work_dir / "merged_raw"
    
    new_profiles = []
    merges_this_iter = 0
    
    for subtree in nontrivial:
        leaves = subtree.get_leaves()
        if len(leaves) < 2:
            continue
        
        logger.info(f"  Merging subtree with {len(leaves)} profiles: {leaves[:3]}...")
        
        # Get RAW profile paths for merge
        profile_paths = {}
        for name in leaves:
            profile_paths[name] = find_profile_path(work_dir, name, prepared=False)
        
        # Create subtree work directory
        subtree_work_dir = work_dir / "subtree_work" / f"subtree_{merges_this_iter}"
        subtree_work_dir.mkdir(parents=True, exist_ok=True)
        
        # Run guided alignment
        try:
            final_path, final_name, merge_records = run_guided_alignment(
                subtree,
                profile_paths,
                subtree_work_dir,
                merge_wrapper,
                generate_merge_id,
                gap_threshold=state.merge_gap_threshold,
                homogeneity_threshold=0.0
            )
            
            if final_path and final_name:
                # Save RAW merged result
                raw_output = merged_raw_dir / f"{final_name}.fasta"
                shutil.copy(final_path, raw_output)
                
                # Prepare filtered + consensus version for hhalign
                prepared_output = merged_dir / f"{final_name}.fasta"
                filtered_cols = prepare_profile_for_hhalign(
                    raw_output, prepared_output, final_name,
                    gap_threshold=state.distance_gap_threshold
                )
                logger.info(f"  Created {final_name}: {filtered_cols} cols after filter")
                
                new_profiles.append(final_name)
                merges_this_iter += len(merge_records)
                
                # Record merge history
                with open(merge_history_file, 'a') as f:
                    for record in merge_records:
                        f.write(f"{record.result_id}\t{record.parent1}\t{record.parent2}\t{state.iteration}\n")
                
                # Update merged_away
                for leaf in leaves:
                    if leaf not in state.merged_away:
                        state.merged_away.append(leaf)
                
        except Exception as e:
            logger.warning(f"  Merge failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Update active profiles
    state.active_profiles = [p for p in active if p not in state.merged_away] + new_profiles
    state.total_merges += merges_this_iter
    
    logger.info(f"Merges this iteration: {merges_this_iter}")
    logger.info(f"Active profiles: {len(state.active_profiles)}")
    
    if merges_this_iter == 0:
        logger.info("No merges performed — converged")
        state.status = "converged"
    else:
        state.status = "processing"
    
    state.save(state_file)
    return 0


@dataclass
class ScoreEntry:
    """Score entry from results file."""
    profile_a: str
    profile_b: str
    score: float
    probability: float
    aligned_cols: int
    coverage: float


def load_scores(filepath: Path) -> Dict[Tuple[str, str], ScoreEntry]:
    """Load scores from TSV file (directional keys)."""
    scores = {}
    if not filepath.exists():
        return scores
    
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                p1, p2 = parts[0], parts[1]
                score = float(parts[2])
                prob = float(parts[3]) if len(parts) > 3 else 0.0
                aligned = int(parts[4]) if len(parts) > 4 else 0
                cov = float(parts[5]) if len(parts) > 5 else 0.0
                
                entry = ScoreEntry(p1, p2, score, prob, aligned, cov)
                key = (p1, p2)
                scores[key] = entry
    return scores


def build_distance_matrix(
    scores: Dict[Tuple[str, str], ScoreEntry],
    profile_names: List[str],
    min_coverage: float = 0.667,
    max_distance: float = 12.0
) -> np.ndarray:
    """
    Build distance matrix from scores using Yuri's algorithm.
    
    Implements hhres2dis_mat with options -n=3 -s=1 -d=2:
    1. Normalize by query self-score
    2. Apply coverage filter
    3. Symmetrize by max
    4. Convert to distance: -log(symmetric)
    """
    n = len(profile_names)
    name_to_idx = {name: i for i, name in enumerate(profile_names)}
    
    # Extract self-scores
    self_scores = {}
    self_aligned = {}
    for name in profile_names:
        key = (name, name)
        if key in scores:
            self_scores[name] = scores[key].score
            self_aligned[name] = scores[key].aligned_cols
        else:
            logger.warning(f"Missing self-score for {name}, using 1.0")
            self_scores[name] = 1.0
            self_aligned[name] = 1
    
    # Build raw score matrix (directional)
    score_matrix = np.zeros((n, n))
    aligned_cols = np.zeros((n, n), dtype=int)
    
    for (p1, p2), entry in scores.items():
        if p1 not in name_to_idx or p2 not in name_to_idx:
            continue
        i, j = name_to_idx[p1], name_to_idx[p2]
        score_matrix[i, j] = entry.score
        aligned_cols[i, j] = entry.aligned_cols
    
    # Step 1: Normalize by query self-score
    normalized = np.zeros((n, n))
    for i in range(n):
        ss = self_scores[profile_names[i]]
        if ss > 0:
            for j in range(n):
                normalized[i, j] = score_matrix[i, j] / ss
        normalized[i, i] = 1.0
    
    # Step 2: Apply coverage filter BEFORE symmetrization
    for i in range(n):
        sa = self_aligned[profile_names[i]]
        if sa > 0:
            for j in range(n):
                if i != j:
                    cov_ij = aligned_cols[i, j] / sa
                    if cov_ij < min_coverage:
                        normalized[i, j] = 0.0
    
    # Step 3: Symmetrize by max
    symmetric = np.maximum(normalized, normalized.T)
    
    # Step 4: Convert to distance
    distance_matrix = np.full((n, n), max_distance)
    np.fill_diagonal(distance_matrix, 0.0)
    
    epsilon = math.exp(-max_distance)
    for i in range(n):
        for j in range(i + 1, n):
            if symmetric[i, j] > epsilon:
                dist = -math.log(symmetric[i, j])
                dist = min(abs(dist), max_distance)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    return distance_matrix


# =============================================================================
# Command: status
# =============================================================================

def cmd_status(args):
    """Show current pipeline status."""
    work_dir = Path(args.work_dir)
    
    state_file = work_dir / "state.json"
    if not state_file.exists():
        logger.error("State file not found. Run 'init' first.")
        return 1
    
    state = PipelineState.load(state_file)
    
    print(f"\n{'='*50}")
    print(f"Pipeline Status: {work_dir}")
    print(f"{'='*50}")
    print(f"  Status: {state.status}")
    print(f"  Iteration: {state.iteration}")
    print(f"  Active profiles: {len(state.active_profiles)}")
    print(f"  Total merges: {state.total_merges}")
    print(f"  Pairs computed: {state.pairs_computed}")
    print(f"\nConfiguration:")
    print(f"  Depth threshold: {state.depth_threshold}")
    print(f"  Depth weight mode: {state.depth_weight_mode}")
    print(f"  Gap threshold (merge): {state.merge_gap_threshold}")
    print(f"  Gap threshold (distance): {state.distance_gap_threshold}")
    print(f"  Coverage threshold: {state.coverage_threshold}")
    print(f"  Max iterations: {state.max_iterations}")
    print(f"{'='*50}\n")
    
    return 0


# =============================================================================
# Command: finalize
# =============================================================================

def cmd_finalize(args):
    """Generate final output."""
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    
    state_file = work_dir / "state.json"
    if not state_file.exists():
        logger.error("State file not found. Run 'init' first.")
        return 1
    
    state = PipelineState.load(state_file)
    
    logger.info(f"=== Finalizing pipeline ===")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Link work directory
    work_link = output_dir / "work"
    if not work_link.exists():
        try:
            work_link.symlink_to(work_dir.resolve())
        except OSError:
            pass
    
    # Generate summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Profile Alignment Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Final profiles: {len(state.active_profiles)}\n")
        f.write(f"Total iterations: {state.iteration}\n")
        f.write(f"Total merges: {state.total_merges}\n")
        f.write(f"Total pairs computed: {state.pairs_computed}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  depth_threshold: {state.depth_threshold}\n")
        f.write(f"  depth_weight_mode: {state.depth_weight_mode}\n")
        f.write(f"  merge_gap_threshold: {state.merge_gap_threshold}\n")
        f.write(f"  distance_gap_threshold: {state.distance_gap_threshold}\n")
        f.write(f"  coverage_threshold: {state.coverage_threshold}\n")
        f.write(f"\nFinal profiles:\n")
        for p in sorted(state.active_profiles):
            f.write(f"  {p}\n")
    
    logger.info(f"Output written to: {output_dir}")
    logger.info(f"Final profiles: {len(state.active_profiles)}")
    
    return 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cluster mode CLI for profile alignment pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {get_version()}')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # init
    p_init = subparsers.add_parser('init', help='Initialize work directory')
    p_init.add_argument('input_dir', type=str, help='Input directory with MSA files')
    p_init.add_argument('work_dir', type=str, help='Work directory')
    p_init.add_argument('--depth-threshold', type=float, default=1.2)
    p_init.add_argument('--depth-weight-mode', type=int, default=2, choices=[0, 1, 2])
    p_init.add_argument('--merge-gap-threshold', type=float, default=0.667)
    p_init.add_argument('--distance-gap-threshold', type=float, default=0.5)
    p_init.add_argument('--length-coverage', type=float, default=0.667,
                        help='Min alignment coverage for distance calculation (default: 0.667)')
    p_init.add_argument('--max-iterations', type=int, default=10)
    
    # gen-queue
    p_queue = subparsers.add_parser('gen-queue', help='Generate task queue')
    p_queue.add_argument('work_dir', type=str, help='Work directory')
    
    # run-task
    p_run_task = subparsers.add_parser("run-task", help="Run one queued hhalign task")
    p_run_task.add_argument("work_dir", type=str, help="Work directory")
    p_run_task.add_argument("task_id", type=str, help="Task ID from current_queue.tsv")

    # process-results
    p_results = subparsers.add_parser('process-results', help='Process results')
    p_results.add_argument('work_dir', type=str, help='Work directory')
    p_results.add_argument('results_file', type=str, nargs='?', help='Results TSV file')
    
    # status
    p_status = subparsers.add_parser('status', help='Show status')
    p_status.add_argument('work_dir', type=str, help='Work directory')
    
    # finalize
    p_final = subparsers.add_parser('finalize', help='Generate final output')
    p_final.add_argument('work_dir', type=str, help='Work directory')
    p_final.add_argument('output_dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "init": cmd_init,
        "gen-queue": cmd_gen_queue,
        "run-task": cmd_run_task,
        "process-results": cmd_process_results,
        "status": cmd_status,
        "finalize": cmd_finalize,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
