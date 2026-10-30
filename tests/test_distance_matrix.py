#!/usr/bin/env python3
"""
Test distance_matrix.py - verify Yuri's algorithm implementation
"""

import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from distance_matrix import (
    build_score_matrix_from_results,
    normalize_scores,
    symmetrize_scores,
    scores_to_distances,
    build_distance_matrix
)


@dataclass
class MockResult:
    """Mock TaskResult for testing."""
    query_name: str
    target_name: str
    score: float
    success: bool = True
    aligned_cols: int = 100  # Default for testing


def test_normalize_scores():
    """Test score normalization by self-scores."""
    print("Testing normalize_scores...")
    
    # Create mock results with known scores
    # A vs A = 100 (self-score)
    # B vs B = 200 (self-score)
    # A vs B = 50
    # B vs A = 80
    results = [
        MockResult("A", "A", 100.0),
        MockResult("B", "B", 200.0),
        MockResult("A", "B", 50.0),
        MockResult("B", "A", 80.0),
    ]
    
    score_matrix = build_score_matrix_from_results(results, ["A", "B"])
    
    # Verify raw scores
    assert score_matrix.scores[0, 0] == 100.0, "A vs A should be 100"
    assert score_matrix.scores[1, 1] == 200.0, "B vs B should be 200"
    assert score_matrix.scores[0, 1] == 50.0, "A vs B should be 50"
    assert score_matrix.scores[1, 0] == 80.0, "B vs A should be 80"
    
    # Normalize by query self-score (mode=3)
    normalized = normalize_scores(score_matrix, normalization_mode=3)
    
    # After normalization:
    # A vs B = 50 / 100 = 0.5
    # B vs A = 80 / 200 = 0.4
    # Diagonal = 1.0
    assert normalized[0, 0] == 1.0, f"Diagonal should be 1.0, got {normalized[0, 0]}"
    assert normalized[1, 1] == 1.0, f"Diagonal should be 1.0, got {normalized[1, 1]}"
    assert abs(normalized[0, 1] - 0.5) < 0.001, f"A vs B should be 0.5, got {normalized[0, 1]}"
    assert abs(normalized[1, 0] - 0.4) < 0.001, f"B vs A should be 0.4, got {normalized[1, 0]}"
    
    print("  normalize_scores: PASSED")


def test_symmetrize_scores():
    """Test score symmetrization."""
    print("Testing symmetrize_scores...")
    
    scores = np.array([
        [1.0, 0.5],
        [0.4, 1.0]
    ])
    
    # Symmetrize by max (mode=1)
    sym_max = symmetrize_scores(scores, symmetrize_mode=1)
    assert sym_max[0, 1] == 0.5, f"Max should be 0.5, got {sym_max[0, 1]}"
    assert sym_max[1, 0] == 0.5, f"Symmetric max should be 0.5, got {sym_max[1, 0]}"
    
    # Symmetrize by min (mode=2)
    sym_min = symmetrize_scores(scores, symmetrize_mode=2)
    assert sym_min[0, 1] == 0.4, f"Min should be 0.4, got {sym_min[0, 1]}"
    assert sym_min[1, 0] == 0.4, f"Symmetric min should be 0.4, got {sym_min[1, 0]}"
    
    print("  symmetrize_scores: PASSED")


def test_scores_to_distances():
    """Test distance transformation."""
    print("Testing scores_to_distances...")
    
    scores = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    
    # Log distance (mode=2): d = -log(s)
    distances = scores_to_distances(scores, distance_mode=2, max_distance=12.0)
    
    # d(A,A) = 0
    assert distances[0, 0] == 0.0, f"Self-distance should be 0, got {distances[0, 0]}"
    
    # d(A,B) = -log(0.5) ≈ 0.693
    expected = -math.log(0.5)
    assert abs(distances[0, 1] - expected) < 0.001, f"Distance should be {expected}, got {distances[0, 1]}"
    
    # Test max_distance for very small scores
    scores_low = np.array([
        [1.0, 0.0001],
        [0.0001, 1.0]
    ])
    distances_low = scores_to_distances(scores_low, distance_mode=2, max_distance=12.0)
    assert distances_low[0, 1] <= 12.0, f"Distance should be capped at 12.0, got {distances_low[0, 1]}"
    
    print("  scores_to_distances: PASSED")


def test_full_pipeline():
    """Test full distance matrix building pipeline."""
    print("Testing full build_distance_matrix...")
    
    # Simulate 3 profiles with known relationships
    # A and B are similar, C is different
    results = [
        # Self-scores
        MockResult("A", "A", 100.0),
        MockResult("B", "B", 100.0),
        MockResult("C", "C", 100.0),
        # A vs B (similar)
        MockResult("A", "B", 80.0),
        MockResult("B", "A", 80.0),
        # A vs C (different)
        MockResult("A", "C", 10.0),
        MockResult("C", "A", 10.0),
        # B vs C (different)
        MockResult("B", "C", 10.0),
        MockResult("C", "B", 10.0),
    ]
    
    distances, names = build_distance_matrix(
        results,
        profile_names=["A", "B", "C"],
        normalization_mode=3,
        symmetrize_mode=1,
        distance_mode=2,
        max_distance=12.0
    )
    
    # Verify shape
    assert distances.shape == (3, 3), f"Expected 3x3, got {distances.shape}"
    
    # Diagonal should be 0
    for i in range(3):
        assert distances[i, i] == 0.0, f"Diagonal[{i}] should be 0, got {distances[i, i]}"
    
    # A-B distance should be smaller than A-C distance
    d_AB = distances[0, 1]
    d_AC = distances[0, 2]
    assert d_AB < d_AC, f"d(A,B)={d_AB} should be < d(A,C)={d_AC}"
    
    # Expected: 
    # A-B normalized = 80/100 = 0.8, d = -log(0.8) ≈ 0.223
    # A-C normalized = 10/100 = 0.1, d = -log(0.1) ≈ 2.303
    expected_AB = -math.log(0.8)
    expected_AC = -math.log(0.1)
    
    assert abs(d_AB - expected_AB) < 0.01, f"d(A,B) should be ~{expected_AB:.3f}, got {d_AB:.3f}"
    assert abs(d_AC - expected_AC) < 0.01, f"d(A,C) should be ~{expected_AC:.3f}, got {d_AC:.3f}"
    
    # Matrix should be symmetric
    assert np.allclose(distances, distances.T), "Distance matrix should be symmetric"
    
    print("  build_distance_matrix: PASSED")
    print(f"  Distance matrix:\n{distances}")


def test_yuri_parameters():
    """Test with Yuri's exact parameters: -n=3 -s=1 -d=2"""
    print("Testing with Yuri's parameters (-n=3 -s=1 -d=2)...")
    
    results = [
        MockResult("profile_A", "profile_A", 500.0),
        MockResult("profile_B", "profile_B", 600.0),
        MockResult("profile_A", "profile_B", 400.0),
        MockResult("profile_B", "profile_A", 450.0),
    ]
    
    distances, names = build_distance_matrix(
        results,
        profile_names=["profile_A", "profile_B"],
        normalization_mode=3,  # -n=3
        symmetrize_mode=1,     # -s=1 (max)
        distance_mode=2,       # -d=2 (log)
        max_distance=12.0
    )
    
    # Normalized scores:
    # A->B = 400/500 = 0.8
    # B->A = 450/600 = 0.75
    # After max symmetrization: max(0.8, 0.75) = 0.8
    # Distance = -log(0.8) ≈ 0.223
    
    expected = -math.log(0.8)
    assert abs(distances[0, 1] - expected) < 0.01, \
        f"Distance should be ~{expected:.4f}, got {distances[0, 1]:.4f}"
    
    print("  Yuri's parameters: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing distance_matrix.py (Yuri's algorithm)")
    print("=" * 60)
    
    test_normalize_scores()
    test_symmetrize_scores()
    test_scores_to_distances()
    test_full_pipeline()
    test_yuri_parameters()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
