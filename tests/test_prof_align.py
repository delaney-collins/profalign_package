#!/usr/bin/env python3
"""
test_prof_align.py - Test the full prof_align pipeline

Runs the pipeline on a subset of profiles and verifies results.

Requires HH-suite (hhalign) installed. Marked as integration tests:
    pytest -m integration tests/test_prof_align.py
"""

import sys
import tempfile
import shutil
import logging
from pathlib import Path

import pytest


from prof_align import PipelineConfig, ProfileAlignmentPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@pytest.mark.integration
def test_small_pipeline(msa_dir: Path, has_hhalign, num_profiles: int = 6):
    """
    Test pipeline on small number of profiles.
    """
    print("\n" + "=" * 70)
    print(f"TEST: Full Pipeline ({num_profiles} profiles)")
    print("=" * 70)
    
    msa_files = sorted(msa_dir.glob("*.fasta"))[:num_profiles]
    
    if len(msa_files) < num_profiles:
        print(f"  SKIPPED: Need at least {num_profiles} MSA files")
        return False
    
    print(f"  Input profiles:")
    for f in msa_files:
        print(f"    - {f.stem}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Copy selected profiles to temp input dir
        input_dir = tmpdir / "input"
        input_dir.mkdir()
        
        for f in msa_files:
            shutil.copy(f, input_dir / f.name)
        
        # Output directory
        output_dir = tmpdir / "output"
        
        # Create config
        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            max_iterations=5,
            depth_threshold=1.2,
            min_profiles_to_stop=1,
        )
        
        # Run pipeline
        print(f"\n  Running pipeline...")
        print(f"    Max iterations: {config.max_iterations}")
        print(f"    Depth threshold: {config.depth_threshold}")
        print()
        
        runner = ProfileAlignmentPipeline(config)
        summary = runner.run()
        
        # Verify results
        print("\n" + "-" * 70)
        print("VERIFICATION")
        print("-" * 70)
        
        # Check summary file exists
        summary_file = output_dir / "summary.txt"
        assert summary_file.exists(), "Summary file should exist"
        print(f"  ✓ Summary file created")
        
        # Check merge history
        history_file = config.history_file
        assert history_file.exists(), "Merge history should exist"
        print(f"  ✓ Merge history created")
        
        # Check distances file
        distances_file = config.distances_file
        assert distances_file.exists(), "Distances file should exist"
        print(f"  ✓ Distances file created")
        
        # Check some merges happened
        assert summary['total_merges'] > 0, "Should have performed some merges"
        print(f"  ✓ Performed {summary['total_merges']} merges")
        
        # Check final profiles reduced
        assert summary['final_profiles'] < summary['initial_profiles'], \
            "Should have fewer profiles after merging"
        print(f"  ✓ Profiles reduced: {summary['initial_profiles']} -> {summary['final_profiles']}")
        
        # Check merged profile files exist
        merged_dir = config.merged_dir
        merged_files = list(merged_dir.glob("M_*.fasta"))
        assert len(merged_files) > 0, "Should have created merged profile files"
        print(f"  ✓ Created {len(merged_files)} merged profile files")
        
        # Print summary
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"  Initial profiles: {summary['initial_profiles']}")
        print(f"  Final profiles: {summary['final_profiles']}")
        print(f"  Iterations: {summary['total_iterations']}")
        print(f"  Total merges: {summary['total_merges']}")
        print(f"  Stop reason: {summary['stop_reason']}")
        print(f"  Time: {summary['elapsed_seconds']:.1f} seconds")
        
        # Show final profiles
        print("\n  Final profiles:")
        for info in summary['final_profile_info']:
            originals = ", ".join(info['input_profiles'][:3])
            if len(info['input_profiles']) > 3:
                originals += f"... (+{len(info['input_profiles'])-3} more)"
            print(f"    {info['name']}: {info['sequences']} seqs, "
                  f"{info['columns']} cols <- [{originals}]")
        
        # Show summary.txt content
        print("\n" + "-" * 70)
        print("GENERATED SUMMARY FILE (excerpt)")
        print("-" * 70)
        with open(summary_file) as f:
            lines = f.readlines()
            for line in lines[:30]:
                print(f"  {line.rstrip()}")
            if len(lines) > 30:
                print(f"  ... ({len(lines) - 30} more lines)")
    
    print("\n  PASSED ✓")
    return True


@pytest.mark.integration
def test_convergence(msa_dir: Path, has_hhalign):
    """
    Test that pipeline converges properly.
    
    Uses profiles known to be related (Cami_1 + Cami_2) plus unrelated ones,
    verifies pipeline merges what it can and stops cleanly.
    """
    print("\n" + "=" * 70)
    print("TEST: Convergence")
    print("=" * 70)
    
    # Pick profiles: Cami_1 + Cami_2 (known to merge) + 2 unrelated
    wanted = ["C25_Cami_1", "C25_Cami_2", "C25_Cam1_1", "C25_Calpl_1"]
    msa_files = []
    for name in wanted:
        f = msa_dir / f"{name}.fasta"
        if f.exists():
            msa_files.append(f)
    
    if len(msa_files) < 3:
        print(f"  SKIPPED: Need at least 3 profiles, found {len(msa_files)}")
        return False
    
    print(f"  Profiles: {[f.stem for f in msa_files]}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        input_dir = tmpdir / "input"
        input_dir.mkdir()
        for f in msa_files:
            shutil.copy(f, input_dir / f.name)
        
        output_dir = tmpdir / "output"
        
        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            max_iterations=10,
            depth_threshold=1.2,
            min_profiles_to_stop=1,
        )
        
        print(f"  Running pipeline...")
        print(f"    Starting with {len(msa_files)} profiles")
        print()
        
        runner = ProfileAlignmentPipeline(config)
        summary = runner.run()
        
        print(f"\n  Results:")
        print(f"    Iterations: {summary['total_iterations']}")
        print(f"    Final profiles: {summary['final_profiles']}")
        print(f"    Total merges: {summary['total_merges']}")
        print(f"    Stop reason: {summary['stop_reason']}")
        
        # Verify: should have merged at least Cami_1 + Cami_2
        assert summary['total_merges'] >= 1, \
            f"Expected at least 1 merge (Cami_1+Cami_2), got {summary['total_merges']}"
        print(f"\n  ✓ At least 1 merge performed")
        
        # Verify: should have converged (stop reason mentions convergence or max iterations)
        assert summary['final_profiles'] < len(msa_files), \
            f"Expected fewer profiles than input ({len(msa_files)}), got {summary['final_profiles']}"
        print(f"  ✓ Profiles reduced: {len(msa_files)} → {summary['final_profiles']}")
        
        # Verify: pipeline stopped cleanly
        assert 'stop_reason' in summary and summary['stop_reason'], \
            "Expected a stop reason"
        print(f"  ✓ Clean stop: {summary['stop_reason']}")
        
    print("\n  PASSED ✓")
    return True


def main():
    print("=" * 70)
    print("PROF_ALIGN FULL PIPELINE TESTS")
    print("=" * 70)
    
    msa_dir = Path(__file__).resolve().parent.parent / "examples" / "msa"
    if not msa_dir.exists():
        print(f"ERROR: {msa_dir} not found")
        print("Run from profalign/ directory")
        sys.exit(1)
    
    results = []
    
    # Test 1: Small pipeline
    results.append(("Small Pipeline", test_small_pipeline(msa_dir, 6)))
    
    # Test 2: Convergence
    results.append(("Convergence", test_convergence(msa_dir)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
