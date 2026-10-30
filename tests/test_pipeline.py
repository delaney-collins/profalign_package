#!/usr/bin/env python3
"""
Integration test for prof_align_v2 with synthetic data and mock merge.

This tests the full pipeline flow without requiring hhalign.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add lib to path


def create_mock_fasta(path: Path, profile_name: str, num_seqs: int = 5, seq_len: int = 100):
    """Create a mock FASTA file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        for i in range(num_seqs):
            seq_id = f"{profile_name}_seq{i+1}"
            # Create a random-ish sequence
            np.random.seed(hash(seq_id) % (2**32))
            seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), seq_len))
            f.write(f">{seq_id}\n{seq}\n")


def create_mock_scores(profile_names: list, families: dict) -> list:
    """
    Create mock HHalign-like score data.
    
    Returns list of (query, target, score, probability, aligned_cols) tuples
    including self-scores and both directions.
    """
    np.random.seed(42)
    SELF_SCORE = 1000.0
    
    profile_to_family = {}
    for family, members in families.items():
        for m in members:
            profile_to_family[m] = family
    
    results = []
    
    # Self-scores first
    for p in profile_names:
        results.append((p, p, SELF_SCORE, 100.0, 100))
    
    # Pairwise scores (both directions)
    for i, p1 in enumerate(profile_names):
        for j, p2 in enumerate(profile_names):
            if i == j:
                continue
            fam1 = profile_to_family.get(p1, "unknown")
            fam2 = profile_to_family.get(p2, "unknown")
            
            if fam1 == fam2:
                # Same family → high score → low distance
                desired_dist = 0.2 + np.random.uniform(0, 0.3)
            else:
                # Different family → low score → high distance
                desired_dist = 5.0 + np.random.uniform(0, 3.0)
            
            # score = self_score * exp(-distance)
            score = SELF_SCORE * np.exp(-desired_dist)
            prob = min(100.0, score / 10)
            results.append((p1, p2, score, prob, 80))
    
    return results


def mock_merge_func(path1: Path, path2: Path, output_path: Path, **kwargs):
    """
    Mock merge function - just concatenates sequences.
    
    In real pipeline, this would use hhalign + merge.
    """
    from formats import read_fasta, write_fasta, Alignment, AlignedSequence
    
    ali1 = read_fasta(path1)
    ali2 = read_fasta(path2)
    
    # Simple concatenation (no real alignment)
    combined_seqs = []
    for seq in ali1.sequences:
        combined_seqs.append(seq)
    for seq in ali2.sequences:
        combined_seqs.append(seq)
    
    merged = Alignment(combined_seqs)
    write_fasta(merged, output_path)
    
    return merged


def test_pipeline_flow():
    """Test the full pipeline flow with mock data."""
    print("=" * 60)
    print("INTEGRATION TEST: prof_align_v2 pipeline flow")
    print("=" * 60)
    
    # Import modules
    from pipeline_utils import ScoreStore, MergeHistory
    from iteration import IterationConfig, IterationCoordinatorV2
    from guided_align import run_guided_alignment
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Setup directories
        input_dir = tmpdir / "input"
        work_dir = tmpdir / "work"
        profiles_dir = work_dir / "profiles"
        merged_dir = work_dir / "merged"
        hhr_dir = work_dir / "hhr"
        
        input_dir.mkdir(parents=True)
        profiles_dir.mkdir(parents=True)
        merged_dir.mkdir(parents=True)
        hhr_dir.mkdir(parents=True)
        
        # Define families
        families = {
            'Cas10': ['Cas10_1', 'Cas10_2', 'Cas10_3', 'Cas10_4'],
            'Cas12': ['Cas12_1', 'Cas12_2', 'Cas12_3'],
            'Cam': ['Cam_1', 'Cam_2'],
        }
        
        profile_names = []
        for members in families.values():
            profile_names.extend(members)
        
        print(f"\nCreating {len(profile_names)} mock profiles...")
        
        # Create mock FASTA files
        for name in profile_names:
            create_mock_fasta(profiles_dir / f"{name}.fasta", name)
        
        print(f"Created profiles: {profile_names}")
        
        # Create mock scores (HHalign-like: self-scores + both directions)
        print("\nCreating mock scores...")
        mock_scores = create_mock_scores(profile_names, families)
        
        distances_file = work_dir / "distances.tsv"
        distances = ScoreStore(distances_file)
        
        for (p1, p2, score, prob, acols) in mock_scores:
            distances.add(p1, p2, score, prob, acols)
        distances.save()
        
        print(f"Created {len(mock_scores)} score entries (incl self-scores + both directions)")
        
        print(f"\nProfiles: {len(profile_names)}, Score entries: {len(mock_scores)}")
        
        # Run iteration
        print("\n" + "=" * 60)
        print("RUNNING ITERATION 1")
        print("=" * 60)
        
        history_file = work_dir / "history.tsv"
        
        config = IterationConfig(
            work_dir=work_dir / "iter_1",
            profiles_dir=profiles_dir,
            distances_file=distances_file,
            merge_history_file=history_file,
            output_profiles_dir=merged_dir,
            hhr_output_dir=hhr_dir / "iter_1",
            depth_threshold=1.2,
            merge_gap_threshold=0.667,
        )
        
        # Monkey-patch the merge function to use our mock
        import iteration
        original_merge = iteration.IterationCoordinatorV2._merge_wrapper
        
        def patched_merge(self, path1, path2, output_path, **kwargs):
            return mock_merge_func(path1, path2, output_path, **kwargs)
        
        iteration.IterationCoordinatorV2._merge_wrapper = patched_merge
        
        try:
            coordinator = IterationCoordinatorV2(config, iteration_num=1)
            result = coordinator.run()
            
            print("\n" + "=" * 60)
            print("ITERATION 1 RESULTS")
            print("=" * 60)
            print(f"Input profiles: {result.num_input_profiles}")
            print(f"Output profiles: {result.num_output_profiles}")
            print(f"Non-trivial subtrees: {result.num_nontrivial_subtrees}")
            print(f"Should continue: {result.should_continue}")
            
            print("\nSubtree results:")
            for sr in result.subtree_results:
                if sr.is_trivial:
                    print(f"  [singleton] {sr.input_profiles[0]}")
                else:
                    print(f"  {sr.input_profiles} -> {sr.output_profile} ({sr.num_merges} merges)")
            
            # Validation
            print("\n" + "=" * 60)
            print("VALIDATION")
            print("=" * 60)
            
            # Expect 3 subtrees (one per family)
            check1 = result.num_output_profiles == 3
            print(f"✓ 3 output profiles: {check1} (got {result.num_output_profiles})")
            
            # Expect all 3 subtrees to be non-trivial
            check2 = result.num_nontrivial_subtrees == 3
            print(f"✓ 3 non-trivial subtrees: {check2} (got {result.num_nontrivial_subtrees})")
            
            # Should continue for another iteration (need to recalculate distances)
            check3 = result.should_continue == True
            print(f"✓ Should continue: {check3}")
            
            # Check merged files exist
            merged_files = list(merged_dir.glob("*.fasta"))
            check4 = len(merged_files) == 3
            print(f"✓ 3 merged files created: {check4} (got {len(merged_files)})")
            
            assert check1 and check2 and check3 and check4, "Some pipeline checks failed"
            print("\n✅ All tests passed!")
                
        finally:
            # Restore original merge function
            iteration.IterationCoordinatorV2._merge_wrapper = original_merge


if __name__ == '__main__':
    test_pipeline_flow()
