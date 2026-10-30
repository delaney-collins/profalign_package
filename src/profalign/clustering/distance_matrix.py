"""
distance_matrix.py - Distance matrix calculation (Yuri's algorithm)

This module implements Yuri's hhres2dis_mat logic:
1. Collect raw scores from all-vs-all comparisons
2. Normalize by self-scores: score[i,j] / self_score[i]
3. Symmetrize: max(score[i,j], score[j,i])
4. Convert to distance: -log(normalized_score)

Reference: Yuri Wolf's hhres2dis_mat script (parec pipeline)
"""

import math
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoreMatrix:
    """
    Raw score matrix before distance transformation.
    
    Stores scores from all-vs-all pairwise comparisons.
    """
    names: List[str]                    # Profile names
    scores: np.ndarray                  # Raw score matrix (n x n)
    aligned_cols: Optional[np.ndarray] = None  # Aligned columns matrix
    
    @property
    def n(self) -> int:
        return len(self.names)
    
    def get_self_scores(self) -> np.ndarray:
        """Get diagonal (self-scores)."""
        return np.diag(self.scores)
    
    def to_dict(self) -> Dict[Tuple[str, str], float]:
        """Convert to dictionary format."""
        result = {}
        for i, name_i in enumerate(self.names):
            for j, name_j in enumerate(self.names):
                result[(name_i, name_j)] = self.scores[i, j]
        return result


def build_score_matrix_from_results(results: List, 
                                     profile_names: List[str] = None) -> ScoreMatrix:
    """
    Build score matrix from TaskResult list.
    
    Args:
        results: List of TaskResult objects (with .score field)
        profile_names: Optional list of profile names (for ordering)
        
    Returns:
        ScoreMatrix with raw scores
    """
    # Collect all unique names
    if profile_names is None:
        names_set = set()
        for r in results:
            names_set.add(r.query_name)
            names_set.add(r.target_name)
        profile_names = sorted(names_set)
    
    n = len(profile_names)
    name_to_idx = {name: i for i, name in enumerate(profile_names)}
    
    # Initialize matrices
    scores = np.zeros((n, n))
    aligned_cols = np.zeros((n, n), dtype=int)
    
    # Fill from results
    for r in results:
        if not r.success:
            continue
        if r.query_name not in name_to_idx or r.target_name not in name_to_idx:
            continue
        
        i = name_to_idx[r.query_name]
        j = name_to_idx[r.target_name]
        
        # Keep best score if multiple results for same pair
        if r.score > scores[i, j]:
            scores[i, j] = r.score
            aligned_cols[i, j] = r.aligned_cols
    
    return ScoreMatrix(
        names=profile_names,
        scores=scores,
        aligned_cols=aligned_cols
    )


def normalize_scores(score_matrix: ScoreMatrix,
                     normalization_mode: int = 3) -> np.ndarray:
    """
    Normalize score matrix by self-scores.
    
    Implements Yuri's -n parameter:
        1: Normalize by min(self_score[i], self_score[j])
        2: Normalize by max(self_score[i], self_score[j])
        3: Normalize by query self-score (self_score[i]) [DEFAULT]
    
    After normalization, diagonal becomes 1.0.
    
    Args:
        score_matrix: ScoreMatrix with raw scores
        normalization_mode: 1=min, 2=max, 3=query (default: 3, matches Yuri's default)
        
    Returns:
        Normalized score matrix (n x n)
    """
    n = score_matrix.n
    scores = score_matrix.scores.copy()
    self_scores = np.diag(scores).copy()
    
    # Avoid division by zero
    self_scores[self_scores <= 0] = 1.0
    
    logger.debug(f"Self-scores: min={self_scores.min():.2f}, max={self_scores.max():.2f}, "
                f"mean={self_scores.mean():.2f}")
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Determine normalization factor based on mode
            if normalization_mode == 1:
                # min of self-scores
                norm_factor = min(self_scores[i], self_scores[j])
            elif normalization_mode == 2:
                # max of self-scores
                norm_factor = max(self_scores[i], self_scores[j])
            else:  # mode 3 (default)
                # query self-score
                norm_factor = self_scores[i]
            
            if norm_factor > 0:
                scores[i, j] /= norm_factor
    
    # Set diagonal to 1.0
    np.fill_diagonal(scores, 1.0)
    
    return scores


def apply_coverage_filter(scores: np.ndarray,
                          score_matrix: ScoreMatrix,
                          min_coverage: float = 0.0,
                          normalization_mode: int = 3) -> np.ndarray:
    """
    Filter scores by alignment coverage (Yuri's -l parameter).
    
    Scores for pairs where normalized coverage < min_coverage are set to 0,
    which will result in maximum distance after log transformation.
    
    Coverage is normalized the same way as scores:
        normalized_coverage = aligned_cols[i,j] / self_aligned_cols[i]
    
    This filters out spurious short matches that may have high scores
    but don't represent true homology.
    
    Args:
        scores: Normalized score matrix (modified in place)
        score_matrix: Original ScoreMatrix with aligned_cols
        min_coverage: Minimum normalized coverage threshold (default: 0.0 = no filter)
        normalization_mode: How to normalize coverage (same as scores)
        
    Returns:
        Filtered score matrix (same array, modified)
    """
    if min_coverage <= 0 or score_matrix.aligned_cols is None:
        return scores
    
    n = score_matrix.n
    aligned_cols = score_matrix.aligned_cols
    self_aligned = np.diag(aligned_cols).astype(float)
    
    # Avoid division by zero
    self_aligned[self_aligned <= 0] = 1.0
    
    filtered_count = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Determine normalization factor (same logic as for scores)
            if normalization_mode == 1:
                norm_factor = min(self_aligned[i], self_aligned[j])
            elif normalization_mode == 2:
                norm_factor = max(self_aligned[i], self_aligned[j])
            else:  # mode 3 (default)
                norm_factor = self_aligned[i]
            
            # Calculate normalized coverage
            if norm_factor > 0:
                normalized_coverage = aligned_cols[i, j] / norm_factor
            else:
                normalized_coverage = 0.0
            
            # Filter: if coverage too low, set score to 0
            if normalized_coverage < min_coverage:
                if scores[i, j] > 0:
                    filtered_count += 1
                scores[i, j] = 0.0
    
    if filtered_count > 0:
        logger.info(f"Coverage filter: {filtered_count} pairs below {min_coverage:.3f} threshold")
    
    return scores


def symmetrize_scores(scores: np.ndarray,
                      symmetrize_mode: int = 1) -> np.ndarray:
    """
    Symmetrize score matrix.
    
    Implements Yuri's -s parameter:
        1: Use maximum of scores[i,j] and scores[j,i] [DEFAULT]
        2: Use minimum of scores[i,j] and scores[j,i]
        0: No symmetrization (keep asymmetric)
    
    Args:
        scores: Score matrix (n x n)
        symmetrize_mode: 1=max (default), 2=min, 0=none
        
    Returns:
        Symmetrized score matrix
    """
    if symmetrize_mode == 0:
        return scores.copy()
    
    scores = scores.copy()
    n = scores.shape[0]
    
    if symmetrize_mode == 1:
        # Maximum
        scores = np.maximum(scores, scores.T)
    elif symmetrize_mode == 2:
        # Minimum
        scores = np.minimum(scores, scores.T)
    
    return scores


def scores_to_distances(scores: np.ndarray,
                        distance_mode: int = 2,
                        max_distance: float = 12.0) -> np.ndarray:
    """
    Convert normalized scores to distances.
    
    Implements Yuri's -d parameter:
        1: Linear distance: 1 - score
        2: Log distance: -log(score) [DEFAULT]
    
    Args:
        scores: Normalized score matrix (values 0-1)
        distance_mode: 1=linear, 2=log (default: 2)
        max_distance: Maximum distance (default: 12.0)
        
    Returns:
        Distance matrix
    """
    n = scores.shape[0]
    distances = np.zeros((n, n))
    epsilon = math.exp(-max_distance)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i, j] = 0.0
                continue
            
            s = scores[i, j]
            
            if distance_mode == 1:
                # Linear: d = 1 - s
                d = 1.0 - s
            else:  # mode 2 (default)
                # Log: d = -log(s)
                if s > epsilon:
                    d = -math.log(s)
                else:
                    d = max_distance
            
            # Ensure non-negative (Yuri uses abs)
            distances[i, j] = abs(d)
            
            # Clamp to max
            if distances[i, j] > max_distance:
                distances[i, j] = max_distance
    
    return distances


def build_distance_matrix(results: List,
                          profile_names: List[str] = None,
                          normalization_mode: int = 3,
                          symmetrize_mode: int = 1,
                          distance_mode: int = 2,
                          max_distance: float = 12.0,
                          min_coverage: float = 0.0) -> Tuple[np.ndarray, List[str]]:
    """
    Build distance matrix from pairwise comparison results.
    
    This is the main entry point that replicates Yuri's hhres2dis_mat:
    1. Build score matrix from results
    2. Normalize by self-scores (-n=3)
    3. Filter by coverage (-l=0.667)
    4. Symmetrize by max (-s=1)
    5. Convert to log distance (-d=2)
    
    IMPORTANT: Results must include self-comparisons (A vs A) for normalization!
    
    Args:
        results: List of TaskResult objects
        profile_names: Optional list of names (for ordering)
        normalization_mode: 1=min, 2=max, 3=query self-score (default: 3)
        symmetrize_mode: 1=max, 2=min, 0=none (default: 1)
        distance_mode: 1=linear, 2=log (default: 2)
        max_distance: Maximum distance value (default: 12.0)
        min_coverage: Minimum normalized alignment coverage (default: 0.0)
                      Pairs with coverage below this are assigned max distance.
                      Yuri's default: 0.667
        
    Returns:
        Tuple of (distance_matrix, profile_names)
    """
    # Step 1: Build raw score matrix
    score_matrix = build_score_matrix_from_results(results, profile_names)
    
    logger.info(f"Building distance matrix for {score_matrix.n} profiles")
    logger.debug(f"Raw scores: min={score_matrix.scores.min():.2f}, "
                f"max={score_matrix.scores.max():.2f}")
    
    # Check self-scores are present
    self_scores = score_matrix.get_self_scores()
    zero_self = np.sum(self_scores <= 0)
    if zero_self > 0:
        logger.warning(f"{zero_self} profiles have zero/missing self-scores! "
                      f"Make sure to include A-vs-A comparisons.")
    
    # Step 2: Normalize by self-scores
    normalized = normalize_scores(score_matrix, normalization_mode)
    
    logger.debug(f"Normalized scores: min={normalized.min():.4f}, "
                f"max={normalized.max():.4f}")
    
    # Step 3: Filter by coverage (Yuri's -l parameter)
    if min_coverage > 0:
        normalized = apply_coverage_filter(
            normalized, score_matrix, min_coverage, normalization_mode
        )
    
    # Step 4: Symmetrize
    symmetric = symmetrize_scores(normalized, symmetrize_mode)
    
    # Step 5: Convert to distances
    distances = scores_to_distances(symmetric, distance_mode, max_distance)
    
    logger.info(f"Distance matrix: min={distances.min():.4f}, "
               f"max={distances.max():.4f}, mean={distances.mean():.4f}")
    
    return distances, score_matrix.names

