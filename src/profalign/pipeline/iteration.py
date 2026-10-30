"""
iteration.py - Iteration coordinator using Yuri's algorithm

This module implements the original prof_align iteration logic:
1. Build distance matrix from all-vs-all comparisons (with normalization!)
2. Build UPGMA tree using hierarchical clustering
3. Cut tree at depth threshold (tree_tips)
4. For each non-trivial subtree: recursively merge using guide tree
5. Result: one profile per subtree
6. Stop when all subtrees are singletons

Reference: Wolf Y.I. prof_align pipeline
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import shutil

logger = logging.getLogger(__name__)


@dataclass
class IterationConfig:
    """Configuration for iteration."""
    work_dir: Path                    # Working directory
    profiles_dir: Path                # Directory with input MSA profiles
    distances_file: Path              # TSV file with pairwise distances
    merge_history_file: Path          # TSV file with merge history
    output_profiles_dir: Path         # Directory for merged profiles
    hhr_output_dir: Path              # Directory for HHR files
    
    # Yuri's parameters
    depth_threshold: float = 1.2      # tree_tips -d parameter (dthr)
    depth_weight_mode: int = 2        # tree_tips -w parameter (0=leaves, 1=TBL, 2=unweighted)
    length_coverage_threshold: float = 0.667  # min alignment length coverage (lthr)
    merge_gap_threshold: float = 0.667  # gap filtering threshold for merge (gcut)
    
    add_consensus: bool = True        # Add consensus for hhalign
    
    def __post_init__(self):
        """Convert strings to Paths and create directories."""
        self.work_dir = Path(self.work_dir)
        self.profiles_dir = Path(self.profiles_dir)
        self.distances_file = Path(self.distances_file)
        self.merge_history_file = Path(self.merge_history_file)
        self.output_profiles_dir = Path(self.output_profiles_dir)
        self.hhr_output_dir = Path(self.hhr_output_dir)
        
        # Create directories
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_profiles_dir.mkdir(parents=True, exist_ok=True)
        self.hhr_output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class SubtreeMergeResult:
    """Result of merging one subtree."""
    input_profiles: List[str]         # Profiles that were merged
    output_profile: str               # Name of resulting profile
    output_path: Path                 # Path to resulting profile
    num_merges: int                   # Number of merge operations
    is_trivial: bool                  # True if singleton (no merge needed)


@dataclass
class IterationResult:
    """Result of one iteration."""
    iteration_num: int
    subtree_results: List[SubtreeMergeResult]
    num_input_profiles: int
    num_output_profiles: int
    num_nontrivial_subtrees: int
    should_continue: bool             # True if there were non-trivial subtrees


class IterationCoordinatorV2:
    """
    Coordinates a single iteration using Yuri's algorithm.
    
    Algorithm:
    1. Load current profiles and distance matrix
    2. Build UPGMA tree
    3. Cut tree at depth threshold -> subtrees
    4. For each non-trivial subtree: guided merge
    5. Output: one profile per subtree
    """
    
    def __init__(self, config: IterationConfig, iteration_num: int = 1):
        self.config = config
        self.iteration_num = iteration_num
        
        # Import dependencies
        from profalign.pipeline.pipeline_utils import (
            generate_merge_id, MergeHistory, ScoreStore
        )
        from profalign.clustering.clustering import (
            hierarchical_clustering,
            linkage_to_newick,
            get_tree_height
        )
        from profalign.alignment.guided_align import (
            scipy_node_to_tree_node,
            calc_tree_weighted_depth,
            extract_subtrees_at_depth,
            get_nontrivial_subtrees,
            run_guided_alignment,
            TreeNode
        )
        from profalign.alignment.merge import merge_two_msas
        from profalign.io.formats import read_fasta, write_fasta
        from profalign.clustering.distance_matrix import build_distance_matrix as build_dist_matrix_from_scores
        
        self.generate_merge_id = generate_merge_id
        self.MergeHistory = MergeHistory
        self.ScoreStore = ScoreStore
        self.hierarchical_clustering = hierarchical_clustering
        self.linkage_to_newick = linkage_to_newick
        self.get_tree_height = get_tree_height
        self.scipy_node_to_tree_node = scipy_node_to_tree_node
        self.calc_tree_weighted_depth = calc_tree_weighted_depth
        self.extract_subtrees_at_depth = extract_subtrees_at_depth
        self.get_nontrivial_subtrees = get_nontrivial_subtrees
        self.run_guided_alignment = run_guided_alignment
        self.TreeNode = TreeNode
        self.merge_two_msas = merge_two_msas
        self.read_fasta = read_fasta
        self.write_fasta = write_fasta
        self.build_dist_matrix_from_scores = build_dist_matrix_from_scores
        
        # Load state
        self.scores = self.ScoreStore(self.config.distances_file)
        self.scores.load()
        
        self.history = self.MergeHistory(self.config.merge_history_file)
        self.history.load()
        
        # Scan current profiles
        self.profile_paths = self._scan_profiles()
    
    def _scan_profiles(self) -> Dict[str, Path]:
        """
        Scan for current active profiles.
        
        Returns dict mapping profile name -> file path.
        Excludes profiles that have been merged away.
        """
        profiles = {}
        
        # Get set of consumed profiles from merge history
        merged_away = set()
        for record in self.history.records.values():
            merged_away.add(record.parent1)
            merged_away.add(record.parent2)
        
        # Scan original profiles directory
        for ext in ['*.fasta', '*.fa', '*.a3m']:
            for f in self.config.profiles_dir.glob(ext):
                name = f.stem
                if name not in merged_away:
                    profiles[name] = f
        
        # Also scan output directory for merged profiles
        for ext in ['*.fasta', '*.fa', '*.a3m']:
            for f in self.config.output_profiles_dir.glob(ext):
                name = f.stem
                if name not in merged_away:
                    profiles[name] = f
        
        return profiles
    
    def get_current_profiles(self) -> List[str]:
        """Get list of current active profile names."""
        return sorted(self.profile_paths.keys())
    
    def build_distance_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build distance matrix from current scores using Yuri's algorithm.
        
        Steps (matching hhres2dis_mat):
        1. Collect all scores including self-scores
        2. Normalize by self-score (-n=3)
        3. Symmetrize by max (-s=1)
        4. Convert to -log distance (-d=2)
        
        Returns:
            Tuple of (matrix, names) where matrix[i,j] is distance between names[i] and names[j]
        """
        names = sorted(self.profile_paths.keys())
        n = len(names)
        
        if n == 0:
            return np.array([]), []
        
        if n == 1:
            return np.array([[0.0]]), names
        
        # Convert scores to TaskResult-like objects for distance_matrix module
        results = self.scores.to_task_results()
        
        # Check if we have self-scores
        self_score_count = sum(1 for r in results if r.query_name == r.target_name)
        if self_score_count < n:
            logger.warning(f"Missing self-scores: have {self_score_count}, need {n}")
        
        # Build distance matrix using Yuri's algorithm
        matrix, ordered_names = self.build_dist_matrix_from_scores(
            results,
            profile_names=names,
            normalization_mode=3,   # Yuri's -n=3: normalize by query self-score
            symmetrize_mode=1,      # Yuri's -s=1: symmetrize by max
            distance_mode=2,        # Yuri's -d=2: -log distance
            max_distance=12.0,
            min_coverage=self.config.length_coverage_threshold  # Yuri's -l=0.667
        )
        
        return matrix, ordered_names
    
    def build_upgma_tree(self, matrix: np.ndarray, names: List[str]):
        """
        Build UPGMA tree from distance matrix.
        
        Args:
            matrix: Distance matrix
            names: Profile names
            
        Returns:
            TreeNode (root of tree)
        """
        from scipy.cluster.hierarchy import to_tree
        
        # Yuri divides all distances by 2 before R hclust (run_R_hclust line 158):
        #   $dd /= 2;  # adjust for hclust() doubling the altitudes
        # This halves tree heights, so depth_threshold=1.2 catches subtrees
        # with inter-cluster distance up to 2.4. Without this, ProfAlign is
        # 2x stricter and misses subtrees that Yuri would merge.
        linkage_matrix = self.hierarchical_clustering(matrix / 2.0, method='upgma')
        scipy_tree = to_tree(linkage_matrix)
        tree = self.scipy_node_to_tree_node(scipy_tree, names)
        
        # Log tree info
        newick = self.linkage_to_newick(linkage_matrix, names)
        logger.debug(f"UPGMA tree: {newick}")
        
        return tree, linkage_matrix
    
    def _merge_wrapper(self, path1: Path, path2: Path, output_path: Path,
                       gap_threshold: float = 0.667,
                       homogeneity_threshold: float = 0.0):
        """
        Wrapper for merge_two_msas that handles paths.
        """
        merged = self.merge_two_msas(
            path1, path2, output_path,
            gap_threshold=gap_threshold,
            homogeneity_threshold=homogeneity_threshold,
            global_alignment=True,
            work_dir=self.config.work_dir / "merge_tmp" / output_path.stem
        )
        return merged
    
    def process_subtree(self, subtree, subtree_idx: int) -> SubtreeMergeResult:
        """
        Process one subtree: merge all profiles following guide tree.
        
        Args:
            subtree: TreeNode (root of subtree)
            subtree_idx: Index for naming
            
        Returns:
            SubtreeMergeResult
        """
        leaves = subtree.get_leaves()
        
        if subtree.is_leaf():
            # Trivial subtree - just copy profile
            profile_name = subtree.name
            profile_path = self.profile_paths[profile_name]
            
            return SubtreeMergeResult(
                input_profiles=[profile_name],
                output_profile=profile_name,
                output_path=profile_path,
                num_merges=0,
                is_trivial=True
            )
        
        # Non-trivial subtree - run guided alignment
        logger.info(f"Processing subtree {subtree_idx}: {len(leaves)} profiles")
        logger.info(f"  Profiles: {leaves}")
        
        subtree_work_dir = self.config.work_dir / f"subtree_{subtree_idx}"
        subtree_work_dir.mkdir(parents=True, exist_ok=True)
        
        # Import MergeRecord type
        from profalign.alignment.guided_align import GuidedMergeRecord
        
        try:
            final_path, final_name, merge_records = self.run_guided_alignment(
                subtree,
                self.profile_paths,
                subtree_work_dir,
                self._merge_wrapper,
                self.generate_merge_id,  # Pass ID generator
                gap_threshold=self.config.merge_gap_threshold,
                homogeneity_threshold=0.0
            )
            
            # Copy final result to output directory
            output_path = self.config.output_profiles_dir / f"{final_name}.fasta"
            shutil.copy(final_path, output_path)
            
            # Record ALL merge steps in history
            for record in merge_records:
                self.history.add(
                    merge_id=record.result_id,
                    parent1=record.parent1,
                    parent2=record.parent2,
                    iteration=self.iteration_num
                )
                # Also update profile_paths for intermediate results
                self.profile_paths[record.result_id] = record.result_path
            
            # Update profile_paths with final output location
            self.profile_paths[final_name] = output_path
            
            ali = self.read_fasta(output_path)
            logger.info(f"  Result: {final_name} ({len(ali)} sequences)")
            
            return SubtreeMergeResult(
                input_profiles=leaves,
                output_profile=final_name,
                output_path=output_path,
                num_merges=len(merge_records),
                is_trivial=False
            )
            
        except Exception as e:
            logger.error(f"  Subtree merge failed: {e}")
            # On failure, return profiles as-is (treat as singletons)
            # This prevents the pipeline from crashing
            return SubtreeMergeResult(
                input_profiles=leaves,
                output_profile=leaves[0],  # Return first profile
                output_path=self.profile_paths[leaves[0]],
                num_merges=0,
                is_trivial=True  # Treat as trivial to avoid infinite loop
            )
    
    def run(self) -> IterationResult:
        """
        Run one iteration of the pipeline.
        
        Returns:
            IterationResult with details of what was done
        """
        logger.info("=" * 60)
        logger.info(f"ITERATION {self.iteration_num} (Yuri's algorithm)")
        logger.info("=" * 60)
        
        profiles = self.get_current_profiles()
        num_input = len(profiles)
        logger.info(f"Current profiles: {num_input}")
        
        if num_input <= 1:
            logger.info("Only one profile remaining - nothing to do")
            return IterationResult(
                iteration_num=self.iteration_num,
                subtree_results=[],
                num_input_profiles=num_input,
                num_output_profiles=num_input,
                num_nontrivial_subtrees=0,
                should_continue=False
            )
        
        # Step 1: Build distance matrix
        logger.info("Building distance matrix...")
        matrix, names = self.build_distance_matrix()
        logger.info(f"  Matrix size: {len(names)}x{len(names)}")
        
        # Step 2: Build UPGMA tree
        logger.info("Building UPGMA tree...")
        tree, linkage_matrix = self.build_upgma_tree(matrix, names)
        tree_height = self.get_tree_height(linkage_matrix)
        logger.info(f"  Tree height: {tree_height:.3f}")
        
        # Step 3: Calculate weighted depths for all nodes
        weight_mode = self.config.depth_weight_mode
        weight_mode_names = {0: "by_leaves", 1: "by_tbl", 2: "unweighted"}
        logger.info(f"Calculating weighted depths (mode={weight_mode_names.get(weight_mode, weight_mode)})...")
        self.calc_tree_weighted_depth(tree, weight_mode)
        
        # Step 4: Extract subtrees at depth threshold
        logger.info(f"Extracting subtrees at depth threshold {self.config.depth_threshold}...")
        subtrees = self.extract_subtrees_at_depth(
            tree, 
            self.config.depth_threshold,
            use_weighted_depth=True
        )
        nontrivial = self.get_nontrivial_subtrees(subtrees)
        
        logger.info(f"  Total subtrees: {len(subtrees)}")
        logger.info(f"  Non-trivial (to merge): {len(nontrivial)}")
        logger.info(f"  Singletons: {len(subtrees) - len(nontrivial)}")
        
        if len(nontrivial) == 0:
            logger.info("All subtrees are singletons - convergence reached")
            return IterationResult(
                iteration_num=self.iteration_num,
                subtree_results=[],
                num_input_profiles=num_input,
                num_output_profiles=num_input,
                num_nontrivial_subtrees=0,
                should_continue=False
            )
        
        # Step 4: Process each subtree
        logger.info("Processing subtrees...")
        results = []
        
        for idx, subtree in enumerate(subtrees, 1):
            result = self.process_subtree(subtree, idx)
            results.append(result)
            
            # Update profile_paths for subsequent subtrees
            # (in case they reference merged profiles - though in tree_tips they shouldn't)
            if not result.is_trivial:
                self.profile_paths[result.output_profile] = result.output_path
        
        # Save history
        self.history.save()
        
        # Count output profiles
        num_output = len(subtrees)
        
        logger.info("-" * 40)
        logger.info(f"Iteration {self.iteration_num} complete:")
        logger.info(f"  Input profiles: {num_input}")
        logger.info(f"  Output profiles: {num_output}")
        logger.info(f"  Merges performed: {sum(r.num_merges for r in results)}")
        
        return IterationResult(
            iteration_num=self.iteration_num,
            subtree_results=results,
            num_input_profiles=num_input,
            num_output_profiles=num_output,
            num_nontrivial_subtrees=len(nontrivial),
            should_continue=len(nontrivial) > 0
        )


def run_iteration(config: IterationConfig, iteration_num: int = 1) -> IterationResult:
    """
    Convenience function to run a single iteration.
    
    Args:
        config: Iteration configuration
        iteration_num: Iteration number for logging
        
    Returns:
        IterationResult
    """
    coordinator = IterationCoordinatorV2(config, iteration_num)
    return coordinator.run()

# Alias for backwards compatibility
IterationCoordinator = IterationCoordinatorV2
