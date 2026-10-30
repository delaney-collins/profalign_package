#!/usr/bin/env python3
"""
test_pipeline_utils.py - Test pipeline utilities

Tests:
1. Merge ID generation (deterministic, symmetric)
2. MergeHistory (save, load, trace originals)
3. ScoreStore (save, load, merge files)
4. Helper functions (find missing pairs)
"""

import tempfile
from pathlib import Path


from pipeline_utils import (
    generate_merge_id,
    MergeHistory,
    ScoreStore,
    find_missing_pairs,
    find_pairs_for_new_profiles,
)


def test_merge_id_generation():
    """Test merge ID generation."""
    print("\n" + "=" * 60)
    print("Test 1: Merge ID Generation")
    print("=" * 60)
    
    # Test determinism
    id1 = generate_merge_id("C25_Calpl_1", "C25_Cami_1")
    id2 = generate_merge_id("C25_Calpl_1", "C25_Cami_1")
    assert id1 == id2, "IDs should be deterministic"
    print(f"  ✓ Deterministic: {id1}")
    
    # Test symmetry
    id3 = generate_merge_id("C25_Cami_1", "C25_Calpl_1")
    assert id1 == id3, "IDs should be symmetric"
    print(f"  ✓ Symmetric: merge(A,B) == merge(B,A)")
    
    # Test uniqueness
    id4 = generate_merge_id("C25_Cas10_1", "C25_Cas10_4")
    assert id1 != id4, "Different pairs should have different IDs"
    print(f"  ✓ Unique: {id1} != {id4}")
    
    # Test format
    assert id1.startswith("M_"), "ID should start with M_"
    assert len(id1) == 14, "ID should be 14 chars (M_ + 12 hex)"
    print(f"  ✓ Format: M_xxxxxxxxxxxx (length={len(id1)})")
    
    # Show some examples
    print("\n  Examples:")
    pairs = [
        ("C25_Calpl_1", "C25_Cami_1"),
        ("C25_Cas10_1", "C25_Cas10_4"),
        ("C25_Cam1_1", "C25_Cam2_1"),
        ("C25_Cas12_B2_1", "C25_Cas12_H1_1"),
    ]
    for p1, p2 in pairs:
        mid = generate_merge_id(p1, p2)
        print(f"    {p1} + {p2} → {mid}")
    
    print("\n  PASSED ✓")


def test_merge_history():
    """Test merge history tracking."""
    print("\n" + "=" * 60)
    print("Test 2: Merge History")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "merge_history.tsv"
        
        # Create and populate
        history = MergeHistory(filepath)
        
        # Iteration 1: merge two original profiles
        mid1 = generate_merge_id("C25_Calpl_1", "C25_Cami_1")
        history.add(mid1, "C25_Calpl_1", "C25_Cami_1", iteration=1)
        print(f"  Added: {mid1} = C25_Calpl_1 + C25_Cami_1 (iter 1)")
        
        mid2 = generate_merge_id("C25_Cas10_1", "C25_Cas10_4")
        history.add(mid2, "C25_Cas10_1", "C25_Cas10_4", iteration=1)
        print(f"  Added: {mid2} = C25_Cas10_1 + C25_Cas10_4 (iter 1)")
        
        # Iteration 2: merge a merged profile with original
        mid3 = generate_merge_id(mid1, "C25_Cam1_1")
        history.add(mid3, mid1, "C25_Cam1_1", iteration=2)
        print(f"  Added: {mid3} = {mid1} + C25_Cam1_1 (iter 2)")
        
        # Check file was written
        assert filepath.exists(), "File should exist"
        print(f"\n  File contents:")
        print("  " + "-" * 50)
        with open(filepath) as f:
            for line in f:
                print(f"  {line.rstrip()}")
        print("  " + "-" * 50)
        
        # Test load
        history2 = MergeHistory(filepath)
        assert len(history2.records) == 3, "Should load 3 records"
        print(f"\n  ✓ Loaded {len(history2.records)} records")
        
        # Test get_parents
        parents = history2.get_parents(mid1)
        assert parents == ("C25_Calpl_1", "C25_Cami_1"), "Parents should match"
        print(f"  ✓ get_parents({mid1}) = {parents}")
        
        # Test get_original_profiles (recursive trace)
        originals = history2.get_original_profiles(mid3)
        expected = ["C25_Calpl_1", "C25_Cami_1", "C25_Cam1_1"]
        assert sorted(originals) == sorted(expected), f"Originals should be {expected}"
        print(f"  ✓ get_original_profiles({mid3}) = {originals}")
        
        # Original profile should return itself
        originals2 = history2.get_original_profiles("C25_Cas10_5")
        assert originals2 == ["C25_Cas10_5"], "Original should return itself"
        print(f"  ✓ get_original_profiles(C25_Cas10_5) = {originals2}")
    
    print("\n  PASSED ✓")


def test_distance_store():
    """Test distance storage."""
    print("\n" + "=" * 60)
    print("Test 3: Distance Store")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "distances.tsv"
        
        # Create and populate
        store = ScoreStore(filepath)
        
        store.add("C25_Calpl_1", "C25_Cami_1", 1.310, probability=92.1, aligned_cols=203)
        store.add("C25_Cam1_1", "C25_Cam2_1", 3.317, probability=45.2, aligned_cols=120)
        store.add("C25_Cas10_1", "C25_Cas10_4", 0.912, probability=99.8, aligned_cols=450)
        
        print(f"  Added 3 distance pairs")
        
        # Check file
        print(f"\n  File contents:")
        print("  " + "-" * 60)
        with open(filepath) as f:
            for line in f:
                print(f"  {line.rstrip()}")
        print("  " + "-" * 60)
        
        # Test directional lookup (ScoreStore stores A→B direction)
        d1 = store.get_score("C25_Calpl_1", "C25_Cami_1")
        assert d1 == 1.310, f"Score should be 1.310, got {d1}"
        print(f"\n  ✓ Directional lookup: get_score(A,B) = {d1}")
        
        # Test symmetric pair detection
        assert store.has_symmetric_pair("C25_Calpl_1", "C25_Cami_1")
        assert store.has_symmetric_pair("C25_Cami_1", "C25_Calpl_1")
        print(f"  ✓ has_symmetric_pair works both ways")
        
        # Test has_pair (directional)
        assert store.has_pair("C25_Calpl_1", "C25_Cami_1")
        assert not store.has_pair("C25_Cami_1", "C25_Calpl_1")  # Directional: only A→B stored
        assert not store.has_pair("C25_Calpl_1", "C25_Cas10_1")
        print(f"  ✓ has_pair works correctly (directional)")
        
        # Test load
        store2 = ScoreStore(filepath)
        assert len(store2.pairs) == 3
        print(f"  ✓ Loaded {len(store2.pairs)} pairs")
        
        # Test get_all_profiles
        profiles = store2.get_all_profiles()
        assert len(profiles) == 6, "Should have 6 unique profiles"
        print(f"  ✓ get_all_profiles: {len(profiles)} profiles")
        
        # Test remove_profile
        store2.remove_profile("C25_Calpl_1")
        assert len(store2.pairs) == 2, "Should have 2 pairs after removal"
        assert not store2.has_pair("C25_Calpl_1", "C25_Cami_1")
        print(f"  ✓ remove_profile: {len(store2.pairs)} pairs remaining")
    
    print("\n  PASSED ✓")


def test_find_missing_pairs():
    """Test helper functions for finding pairs to calculate."""
    print("\n" + "=" * 60)
    print("Test 4: Find Missing/New Pairs")
    print("=" * 60)
    
    # Create store with some existing directional pairs
    store = ScoreStore()
    store.add("P001", "P002", 1.0, 80.0, 100, append_to_file=False)
    store.add("P001", "P003", 1.5, 75.0, 95, append_to_file=False)
    store.add("P002", "P003", 2.0, 70.0, 90, append_to_file=False)
    
    profiles = ["P001", "P002", "P003", "P004"]
    
    # find_missing_pairs returns ALL missing directional pairs + self
    # 4 profiles → 16 total directional pairs (incl self)
    # 3 stored → 13 missing
    missing = find_missing_pairs(profiles, store)
    assert len(missing) == 13, f"Should have 13 missing directional pairs, got {len(missing)}"
    print(f"  ✓ find_missing_pairs: {len(missing)} missing (directional + self)")
    
    # Test find_pairs_for_new_profiles
    # Scenario: P005 is newly created, needs distances to all others
    new_profiles = ["P005"]
    all_profiles = ["P001", "P002", "P003", "P004", "P005"]
    
    new_pairs = find_pairs_for_new_profiles(new_profiles, all_profiles, store)
    assert len(new_pairs) == 4, f"P005 needs 4 pairs, got {len(new_pairs)}"
    print(f"  ✓ find_pairs_for_new_profiles (1 new): {new_pairs}")
    
    # Scenario: Two new profiles P005 and P006
    new_profiles2 = ["P005", "P006"]
    all_profiles2 = ["P001", "P002", "P003", "P005", "P006"]
    
    new_pairs2 = find_pairs_for_new_profiles(new_profiles2, all_profiles2, store)
    # P005 needs: P001, P002, P003, P006 (4 pairs)
    # P006 needs: P001, P002, P003 (3 pairs, P005-P006 already counted)
    # Total: 7 pairs
    assert len(new_pairs2) == 7, f"Should have 7 pairs, got {len(new_pairs2)}"
    print(f"  ✓ find_pairs_for_new_profiles (2 new): {len(new_pairs2)} pairs")
    for p in new_pairs2:
        print(f"      {p[0]} vs {p[1]}")
    
    print("\n  PASSED ✓")


def main():
    print("=" * 60)
    print("Pipeline Utilities Tests")
    print("=" * 60)
    
    test_merge_id_generation()
    test_merge_history()
    test_distance_store()
    test_find_missing_pairs()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
