#!/usr/bin/env python3
"""
prof_align.py - Main pipeline for iterative profile-guided MSA alignment

This is the main entry point for the prof_align pipeline.
Implements Yuri Wolf's algorithm:
1. All-vs-all pairwise comparison (hhalign)
2. Build UPGMA tree from distance matrix
3. Cut tree at depth threshold (tree_tips)
4. Merge profiles within each subtree following guide tree
5. Repeat until all subtrees are singletons

Usage:
    python prof_align.py <msa_dir> <output_dir> [options]

Example:
    python prof_align.py ../data/msa ./results --max-iterations 5

Reference: Wolf Y.I. prof_align pipeline
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Setup path for imports


from .pipeline.pipeline_utils import (
    ScoreStore, MergeHistory, find_missing_pairs
)
from .pipeline.task_queue import (
    TaskQueue, run_task_queue_sequential, results_to_score_store
)
from .pipeline.iteration import (
    IterationConfig, IterationCoordinator, IterationResult
)
from .io.formats import read_fasta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for the full pipeline."""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        max_iterations: int = 10,
        depth_threshold: float = 1.2,       # tree_tips -d parameter
        depth_weight_mode: int = 2,         # tree_tips -w parameter (0=leaves, 1=tbl, 2=unweighted)
        length_coverage: float = 0.667,      # min alignment length coverage
        merge_gap_threshold: float = 0.667,  # gap filtering for merge
        distance_gap_threshold: float = 0.5, # gap filtering before distance calc (Yuri's -gcon)
        add_consensus: bool = True,
        min_profiles_to_stop: int = 1,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.depth_threshold = depth_threshold
        self.depth_weight_mode = depth_weight_mode
        self.length_coverage = length_coverage
        self.merge_gap_threshold = merge_gap_threshold
        self.distance_gap_threshold = distance_gap_threshold
        self.add_consensus = add_consensus
        self.min_profiles_to_stop = min_profiles_to_stop
        
        # Derived paths
        self.work_dir = self.output_dir / "work"
        self.profiles_dir = self.work_dir / "profiles"
        self.merged_dir = self.work_dir / "merged"
        self.hhr_dir = self.work_dir / "hhr"
        self.distances_file = self.work_dir / "distances.tsv"
        self.history_file = self.work_dir / "merge_history.tsv"
        self.log_file = self.output_dir / "pipeline.log"
        
    def create_directories(self):
        """Create all necessary directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        self.hhr_dir.mkdir(parents=True, exist_ok=True)


class ProfileAlignmentPipeline:
    """
    Main pipeline class implementing Yuri's algorithm.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.iteration_results: List[IterationResult] = []
        self.start_time = None
        self.end_time = None
        
    def initialize(self) -> int:
        """
        Initialize the pipeline.
        
        1. Create directories
        2. Copy input MSAs to working directory
        3. Calculate initial pairwise distances
        
        Returns:
            Number of initial profiles
        """
        logger.info("=" * 70)
        logger.info("INITIALIZING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Input directory: {self.config.input_dir}")
        logger.info(f"Output directory: {self.config.output_dir}")
        
        # Create directories
        self.config.create_directories()
        
        # Setup file logging
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
        
        # Copy input MSAs
        logger.info("")
        logger.info("Copying input MSAs...")
        input_files = list(self.config.input_dir.glob("*.fasta")) + \
                      list(self.config.input_dir.glob("*.fa")) + \
                      list(self.config.input_dir.glob("*.a3m"))
        input_files.sort(key=lambda f: f.stem)  # Deterministic order
        
        profile_paths = {}
        for f in input_files:
            dest = self.config.profiles_dir / f"{f.stem}.fasta"
            if not dest.exists():
                shutil.copy(f, dest)
            profile_paths[f.stem] = dest
            
        logger.info(f"  Copied {len(profile_paths)} profiles")
        
        # Calculate initial pairwise scores (for Yuri-style distance calculation)
        logger.info("")
        logger.info("Calculating initial pairwise scores...")
        
        scores = ScoreStore(self.config.distances_file)
        scores.load()
        
        # Find missing pairs (including self-comparisons for normalization!)
        profile_names = sorted(profile_paths.keys())
        pairs = find_missing_pairs(profile_names, scores, include_self=True)
        
        logger.info(f"  Need to calculate {len(pairs)} pairs (including self-comparisons)")
        
        if pairs:
            # Create task queue
            queue = TaskQueue()
            queue.add_tasks_from_pairs(
                pairs, 
                profile_paths,
                self.config.hhr_dir / "initial"
            )
            
            # Run tasks (coverage filtering is applied later in distance_matrix.py)
            results = run_task_queue_sequential(
                queue,
                add_consensus=self.config.add_consensus,
                gap_threshold=self.config.distance_gap_threshold
            )
            
            # Save results (raw scores, not distances)
            results_to_score_store(results, scores)
            scores.save()
            
            successful = sum(1 for r in results if r.success)
            logger.info(f"Calculated {successful}/{len(pairs)} scores")
        
        return len(profile_paths)
    
    def run_iteration(self, iteration_num: int) -> IterationResult:
        """
        Run a single iteration using Yuri's algorithm.
        
        1. Build UPGMA tree from distances
        2. Cut tree at depth threshold
        3. For each non-trivial subtree: guided merge
        """
        
        # Create iteration config
        iter_config = IterationConfig(
            work_dir=self.config.work_dir / f"iter_{iteration_num}",
            profiles_dir=self.config.profiles_dir,
            distances_file=self.config.distances_file,
            merge_history_file=self.config.history_file,
            output_profiles_dir=self.config.merged_dir,
            hhr_output_dir=self.config.hhr_dir / f"iter_{iteration_num}",
            depth_threshold=self.config.depth_threshold,
            depth_weight_mode=self.config.depth_weight_mode,
            length_coverage_threshold=self.config.length_coverage,
            merge_gap_threshold=self.config.merge_gap_threshold,
            add_consensus=self.config.add_consensus,
        )
        
        coordinator = IterationCoordinator(iter_config, iteration_num)
        result = coordinator.run()
        
        # After merging, we need to calculate scores for new profiles
        if result.num_nontrivial_subtrees > 0:
            self._update_scores_after_iteration(result)
        
        return result
    
    def _update_scores_after_iteration(self, result: IterationResult):
        """
        Calculate scores for newly merged profiles.
        
        New profiles need scores vs all current profiles (both directions)
        plus self-scores for normalization.
        """
        # Get new profile names
        new_profiles = []
        for sr in result.subtree_results:
            if not sr.is_trivial:
                new_profiles.append(sr.output_profile)
        
        if not new_profiles:
            return
        
        logger.info(f"Calculating scores for {len(new_profiles)} new profiles...")
        
        # Load current scores
        scores = ScoreStore(self.config.distances_file)
        scores.load()
        
        # Get all current active profiles
        # This includes: new merged profiles + singletons from this iteration
        all_current = []
        for sr in result.subtree_results:
            all_current.append(sr.output_profile)
        
        # Get paths for all profiles
        profile_paths = {}
        for prof in all_current:
            # Check merged dir first
            path = self.config.merged_dir / f"{prof}.fasta"
            if path.exists():
                profile_paths[prof] = path
            else:
                # Check original profiles dir
                path = self.config.profiles_dir / f"{prof}.fasta"
                if path.exists():
                    profile_paths[prof] = path
        
        # Find pairs that need calculation
        # For new profiles: self-comparison + all vs others (both directions)
        pairs_needed = []
        for new_prof in new_profiles:
            # Self-comparison (for normalization)
            if not scores.has_pair(new_prof, new_prof):
                pairs_needed.append((new_prof, new_prof))
            
            # New vs all others (both directions for asymmetric scores)
            for other_prof in all_current:
                if new_prof != other_prof:
                    if not scores.has_pair(new_prof, other_prof):
                        pairs_needed.append((new_prof, other_prof))
                    if not scores.has_pair(other_prof, new_prof):
                        pairs_needed.append((other_prof, new_prof))
        
        if not pairs_needed:
            logger.info("  No new pairs to calculate")
            return
        
        logger.info(f"  Need to calculate {len(pairs_needed)} pairs (including self)")
        
        # Create task queue
        queue = TaskQueue()
        queue.add_tasks_from_pairs(
            pairs_needed,
            profile_paths,
            self.config.hhr_dir / f"iter_{result.iteration_num}_scores"
        )
        
        # Run tasks (coverage filtering is applied later in distance_matrix.py)
        results = run_task_queue_sequential(
            queue,
            add_consensus=self.config.add_consensus,
            gap_threshold=self.config.distance_gap_threshold
        )
        
        # Save results
        results_to_score_store(results, scores)
        scores.save()
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"  Calculated {successful}/{len(pairs_needed)} scores")
    
    def should_stop(self, result: IterationResult, iteration_num: int) -> tuple:
        """
        Check if pipeline should stop.
        
        Yuri's stopping criteria:
        1. All subtrees are singletons (no non-trivial subtrees)
        2. Reached max iterations
        3. Only one profile remaining
        
        Returns:
            (should_stop: bool, reason: str)
        """
        # No non-trivial subtrees = convergence
        if result.num_nontrivial_subtrees == 0:
            return True, "Convergence: all subtrees are singletons"
        
        # Only one profile remaining
        if result.num_output_profiles <= self.config.min_profiles_to_stop:
            return True, f"Reached minimum profiles ({result.num_output_profiles})"
        
        # Reached max iterations
        if iteration_num >= self.config.max_iterations:
            return True, f"Reached maximum iterations ({self.config.max_iterations})"
        
        # Check should_continue from result
        if not result.should_continue:
            return True, "No more merges possible"
        
        return False, ""
    
    def run(self) -> dict:
        """
        Run the full pipeline.
        
        Returns:
            Summary dictionary with results
        """
        self.start_time = time.time()
        
        # Initialize
        initial_profiles = self.initialize()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("STARTING ITERATIVE MERGING (Yuri's Algorithm)")
        logger.info("=" * 70)
        logger.info(f"Max iterations: {self.config.max_iterations}")
        logger.info(f"Depth threshold: {self.config.depth_threshold}")
        weight_mode_names = {0: "by_leaves", 1: "by_tbl", 2: "unweighted"}
        logger.info(f"Depth weight mode: {self.config.depth_weight_mode} ({weight_mode_names.get(self.config.depth_weight_mode, 'unknown')})")
        logger.info(f"Gap threshold (merge): {self.config.merge_gap_threshold}")
        logger.info(f"Gap threshold (distance): {self.config.distance_gap_threshold}")
        
        # Iteration loop
        iteration_num = 0
        stop_reason = ""
        
        while True:
            iteration_num += 1
            
            logger.info("")
            result = self.run_iteration(iteration_num)
            self.iteration_results.append(result)
            
            # Check stopping criteria
            should_stop, stop_reason = self.should_stop(result, iteration_num)
            if should_stop:
                logger.info(f"Stopping: {stop_reason}")
                break
        
        self.end_time = time.time()
        
        # Generate summary
        summary = self.generate_summary(stop_reason, initial_profiles)
        self.write_summary(summary)
        
        return summary
    
    def generate_summary(self, stop_reason: str, initial_profiles: int) -> dict:
        """Generate pipeline summary."""
        
        total_merges = sum(
            sum(sr.num_merges for sr in r.subtree_results)
            for r in self.iteration_results
        )
        
        final_profiles = self.iteration_results[-1].num_output_profiles if self.iteration_results else 0
        
        # Load merge history for tracing
        history = MergeHistory(self.config.history_file)
        history.load()
        
        # Get final profile info
        # Find the last iteration with actual subtree results
        # (convergence iteration returns empty subtree_results)
        final_profile_info = []
        
        result_with_profiles = None
        for r in reversed(self.iteration_results):
            if r.subtree_results:
                result_with_profiles = r
                break
        
        if result_with_profiles:
            for sr in result_with_profiles.subtree_results:
                path = sr.output_path
                if path and path.exists():
                    ali = read_fasta(path)
                    profile_name = sr.output_profile
                    # Check MergeHistory to determine if profile was ever created by merging
                    is_merged = history.get(profile_name) is not None
                    # Get original profiles from history (recursive trace back)
                    if is_merged:
                        original_profiles = history.get_original_profiles(profile_name)
                    else:
                        original_profiles = [profile_name]
                    final_profile_info.append({
                        'name': profile_name,
                        'sequences': len(ali.sequences),
                        'columns': ali.get_length(),
                        'input_profiles': original_profiles,
                        'num_originals': len(original_profiles),
                        'is_merged': is_merged,
                    })
        else:
            # Fallback: collect from file system if no iteration had results
            # This handles edge case of single-iteration convergence
            for fasta_path in sorted(self.config.profiles_dir.glob("*.fasta")):
                ali = read_fasta(fasta_path)
                profile_name = fasta_path.stem
                # Check MergeHistory even in fallback
                is_merged = history.get(profile_name) is not None
                if is_merged:
                    original_profiles = history.get_original_profiles(profile_name)
                else:
                    original_profiles = [profile_name]
                final_profile_info.append({
                    'name': profile_name,
                    'sequences': len(ali.sequences),
                    'columns': ali.get_length(),
                    'input_profiles': original_profiles,
                    'num_originals': len(original_profiles),
                    'is_merged': is_merged,
                })
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'Yuri (UPGMA + tree_tips + guided_align)',
            'input_dir': str(self.config.input_dir),
            'output_dir': str(self.config.output_dir),
            'parameters': {
                'depth_threshold': self.config.depth_threshold,
                'depth_weight_mode': self.config.depth_weight_mode,
                'merge_gap_threshold': self.config.merge_gap_threshold,
                'distance_gap_threshold': self.config.distance_gap_threshold,
                'length_coverage': self.config.length_coverage,
                'max_iterations': self.config.max_iterations,
            },
            'initial_profiles': initial_profiles,
            'final_profiles': final_profiles,
            'total_iterations': len(self.iteration_results),
            'total_merges': total_merges,
            'stop_reason': stop_reason,
            'elapsed_seconds': self.end_time - self.start_time if self.end_time else 0,
            'iteration_details': [
                {
                    'iteration': r.iteration_num,
                    'input_profiles': r.num_input_profiles,
                    'output_profiles': r.num_output_profiles,
                    'nontrivial_subtrees': r.num_nontrivial_subtrees,
                    'subtrees': [
                        {
                            'profiles': sr.input_profiles,
                            'result': sr.output_profile,
                            'merges': sr.num_merges,
                            'is_trivial': sr.is_trivial,
                        }
                        for sr in r.subtree_results
                    ]
                }
                for r in self.iteration_results
            ],
            'final_profile_info': final_profile_info,
        }
        
        return summary
    
    def write_summary(self, summary: dict):
        """Write summary to file."""
        summary_file = self.config.output_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("PROFILE ALIGNMENT PIPELINE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Algorithm: {summary['algorithm']}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Input: {summary['input_dir']}\n")
            f.write(f"Output: {summary['output_dir']}\n")
            f.write("\n")
            
            f.write("Parameters:\n")
            for k, v in summary['parameters'].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
            
            f.write("Results:\n")
            f.write(f"  Initial profiles: {summary['initial_profiles']}\n")
            f.write(f"  Final profiles: {summary['final_profiles']}\n")
            f.write(f"  Total iterations: {summary['total_iterations']}\n")
            f.write(f"  Total merges: {summary['total_merges']}\n")
            f.write(f"  Stop reason: {summary['stop_reason']}\n")
            f.write(f"  Elapsed time: {summary['elapsed_seconds']:.1f} seconds\n")
            f.write("\n")
            
            f.write("Iterations:\n")
            for detail in summary['iteration_details']:
                f.write(f"\n  Iteration {detail['iteration']}:\n")
                f.write(f"    Input: {detail['input_profiles']} profiles\n")
                f.write(f"    Output: {detail['output_profiles']} profiles\n")
                f.write(f"    Non-trivial subtrees: {detail['nontrivial_subtrees']}\n")
                
                for st in detail['subtrees']:
                    if st['is_trivial']:
                        f.write(f"    - [singleton] {st['profiles'][0]}\n")
                    else:
                        f.write(f"    - {st['profiles']} -> {st['result']} ({st['merges']} merges)\n")
            
            f.write("\n")
            f.write("Final Profiles:\n")
            for info in summary['final_profile_info']:
                status = "merged" if info['is_merged'] else "original"
                f.write(f"  {info['name']}: {info['sequences']} seqs, {info['columns']} cols ({status})\n")
                if info['is_merged']:
                    f.write(f"    From: {info['input_profiles']}\n")
        
        logger.info(f"Summary written to: {summary_file}")
        
        # Also log summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Initial profiles: {summary['initial_profiles']}")
        logger.info(f"Final profiles: {summary['final_profiles']}")
        logger.info(f"Total iterations: {summary['total_iterations']}")
        logger.info(f"Total merges: {summary['total_merges']}")
        logger.info(f"Elapsed time: {summary['elapsed_seconds']:.1f} seconds")


def get_version():
    """Read version from VERSION file."""
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "unknown"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile alignment pipeline (Yuri's algorithm)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic run
    python prof_align.py ../data/msa ./results
    
    # Conservative merging (lower depth threshold)
    python prof_align.py ../data/msa ./results --depth-threshold 1.0 --max-iterations 5
    
    # More aggressive merging (higher depth threshold)
    python prof_align.py ../data/msa ./results --depth-threshold 2.0
"""
    )
    
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {get_version()}')
    
    parser.add_argument('input_dir', type=Path,
                        help='Directory containing input MSA files (FASTA format)')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for results')
    
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum number of iterations (default: 10)')
    parser.add_argument('--depth-threshold', type=float, default=1.2,
                        help='Tree depth threshold for subtree cutting (default: 1.2)')
    parser.add_argument('--depth-weight-mode', type=int, default=2, choices=[0, 1, 2],
                        help='Weighting mode for tree depth calculation (default: 2). '
                             '0=by number of leaves, 1=by total branch length, 2=unweighted. '
                             'Corresponds to Yuri\'s -w parameter.')
    parser.add_argument('--merge-gap-threshold', type=float, default=0.667,
                        help='Gap threshold for column filtering during merge (default: 0.667)')
    parser.add_argument('--distance-gap-threshold', type=float, default=0.5,
                        help='Gap threshold for column filtering before distance calculation (default: 0.5). '
                             'Columns with gap fraction > this are removed before hhalign. '
                             'Corresponds to Yuri\'s -gcon parameter. Use 1.0 to disable.')
    parser.add_argument('--length-coverage', type=float, default=0.667,
                        help='Minimum alignment coverage for distance calculation (default: 0.667). '
                             'Pairs with coverage below this are treated as unrelated (max distance). '
                             'Corresponds to Yuri\'s -l parameter.')
    parser.add_argument('--no-consensus', action='store_true',
                        help='Do not add consensus sequences for hhalign')
    parser.add_argument('--min-profiles', type=int, default=1,
                        help='Stop when this many profiles remain (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    # Create config
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        depth_threshold=args.depth_threshold,
        depth_weight_mode=args.depth_weight_mode,
        merge_gap_threshold=args.merge_gap_threshold,
        distance_gap_threshold=args.distance_gap_threshold,
        length_coverage=args.length_coverage,
        add_consensus=not args.no_consensus,
        min_profiles_to_stop=args.min_profiles,
    )
    
    # Run pipeline
    pipeline = ProfileAlignmentPipeline(config)
    
    try:
        summary = pipeline.run()
        return 0
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
