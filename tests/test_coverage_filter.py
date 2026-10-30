#!/usr/bin/env python3
"""
Test coverage filter functionality.

Tests the coverage calculation and filtering logic in distance calculation.
"""

import math
from pathlib import Path

from dataclasses import dataclass


def test_coverage_calculation():
    """
    Test coverage calculation logic.
    
    Coverage = min(query_coverage, template_coverage)
    where:
      query_coverage = aligned_cols / query_length
      template_coverage = aligned_cols / template_length
    """
    print("TEST: Coverage calculation")
    
    # Test case 1: Equal length profiles, full alignment
    query_len = 100
    template_len = 100
    aligned_cols = 100
    
    query_cov = aligned_cols / query_len
    template_cov = aligned_cols / template_len
    coverage = min(query_cov, template_cov)
    
    assert coverage == 1.0, f"Expected 1.0, got {coverage}"
    print(f"  Full alignment: coverage = {coverage:.3f} ✓")
    
    # Test case 2: Partial alignment
    aligned_cols = 50
    query_cov = aligned_cols / query_len
    template_cov = aligned_cols / template_len
    coverage = min(query_cov, template_cov)
    
    assert coverage == 0.5, f"Expected 0.5, got {coverage}"
    print(f"  50% alignment: coverage = {coverage:.3f} ✓")
    
    # Test case 3: Different length profiles
    query_len = 100
    template_len = 200
    aligned_cols = 80
    
    query_cov = aligned_cols / query_len  # 0.8
    template_cov = aligned_cols / template_len  # 0.4
    coverage = min(query_cov, template_cov)
    
    assert coverage == 0.4, f"Expected 0.4, got {coverage}"
    print(f"  Asymmetric (100 vs 200, 80 aligned): coverage = {coverage:.3f} ✓")
    
    # Test case 4: Short hit on long profiles
    query_len = 500
    template_len = 600
    aligned_cols = 50
    
    query_cov = aligned_cols / query_len  # 0.1
    template_cov = aligned_cols / template_len  # 0.083
    coverage = min(query_cov, template_cov)
    
    assert abs(coverage - 0.0833) < 0.001, f"Expected ~0.083, got {coverage}"
    print(f"  Short hit (50/500 vs 50/600): coverage = {coverage:.3f} ✓")
    
    print("✓ PASSED\n")


def test_coverage_filter():
    """
    Test that coverage filter correctly rejects low-coverage pairs.
    """
    print("TEST: Coverage filter logic")
    
    min_coverage = 0.667  # Yuri's default
    max_distance = 12.0
    
    def calculate_distance(probability, coverage, min_coverage, max_distance):
        """Replicate distance calculation logic from task_queue.py"""
        if coverage < min_coverage:
            return max_distance  # Low coverage -> max distance
        elif probability > 0:
            prob_clamped = min(probability, 100.0)
            distance = -math.log(prob_clamped / 100.0)
            return max(0.0, min(distance, max_distance))
        else:
            return max_distance
    
    # Test case 1: High probability, high coverage -> low distance
    prob, cov = 99.0, 0.9
    dist = calculate_distance(prob, cov, min_coverage, max_distance)
    expected = -math.log(99.0 / 100.0)  # ~0.01
    assert abs(dist - expected) < 0.001, f"Expected {expected:.4f}, got {dist:.4f}"
    print(f"  High prob (99%), high cov (0.9): distance = {dist:.4f} ✓")
    
    # Test case 2: High probability, LOW coverage -> MAX distance (filtered!)
    prob, cov = 99.0, 0.3  # Coverage below threshold
    dist = calculate_distance(prob, cov, min_coverage, max_distance)
    assert dist == max_distance, f"Expected {max_distance}, got {dist}"
    print(f"  High prob (99%), low cov (0.3): distance = {dist:.4f} (FILTERED) ✓")
    
    # Test case 3: Low probability, high coverage -> high distance
    prob, cov = 1.0, 0.8
    dist = calculate_distance(prob, cov, min_coverage, max_distance)
    expected = -math.log(1.0 / 100.0)  # ~4.6
    assert abs(dist - expected) < 0.001, f"Expected {expected:.4f}, got {dist:.4f}"
    print(f"  Low prob (1%), high cov (0.8): distance = {dist:.4f} ✓")
    
    # Test case 4: Borderline coverage (exactly at threshold)
    prob, cov = 50.0, 0.667
    dist = calculate_distance(prob, cov, min_coverage, max_distance)
    expected = -math.log(50.0 / 100.0)  # ~0.69
    assert abs(dist - expected) < 0.001, f"Expected {expected:.4f}, got {dist:.4f}"
    print(f"  Medium prob (50%), borderline cov (0.667): distance = {dist:.4f} ✓")
    
    # Test case 5: Just below threshold
    prob, cov = 50.0, 0.666
    dist = calculate_distance(prob, cov, min_coverage, max_distance)
    assert dist == max_distance, f"Expected {max_distance}, got {dist}"
    print(f"  Medium prob (50%), below threshold (0.666): distance = {dist:.4f} (FILTERED) ✓")
    
    print("✓ PASSED\n")


def test_distance_pair_coverage():
    """
    Test ScorePair with coverage field.
    """
    print("TEST: ScorePair with coverage")
    
    from pipeline_utils import ScorePair
    
    # Create pair with coverage
    pair = ScorePair(
        profile_a="prof1",
        profile_b="prof2",
        score=1.5,
        probability=50.0,
        aligned_cols=100,
        coverage=0.75
    )
    
    # Test to_tsv_row
    row = pair.to_tsv_row()
    assert "0.7500" in row, f"Coverage not in row: {row}"
    print(f"  to_tsv_row: {row} ✓")
    
    # Test from_tsv_row (with coverage)
    parts = row.split('\t')
    pair2 = ScorePair.from_tsv_row(parts)
    assert pair2.coverage == 0.75, f"Expected 0.75, got {pair2.coverage}"
    print(f"  from_tsv_row: coverage = {pair2.coverage} ✓")
    
    # Test backward compatibility (without coverage)
    old_row = ["prof1", "prof2", "1.5", "50.0", "100"]
    pair3 = ScorePair.from_tsv_row(old_row)
    assert pair3.coverage == 0.0, f"Expected 0.0 for old format, got {pair3.coverage}"
    print(f"  Backward compatible (no coverage): coverage = {pair3.coverage} ✓")
    
    print("✓ PASSED\n")


def test_task_result_coverage():
    """
    Test TaskResult with coverage field.
    """
    print("TEST: TaskResult with coverage")
    
    from task_queue import TaskResult
    
    result = TaskResult(
        query_name="query",
        target_name="target",
        score=1.0,
        probability=60.0,
        aligned_cols=150,
        coverage=0.8,
        success=True
    )
    
    assert result.coverage == 0.8
    print(f"  TaskResult.coverage = {result.coverage} ✓")
    
    print("✓ PASSED\n")


def main():
    print("=" * 60)
    print("Coverage Filter Tests")
    print("=" * 60)
    print()
    
    test_coverage_calculation()
    test_coverage_filter()
    test_distance_pair_coverage()
    test_task_result_coverage()
    
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
