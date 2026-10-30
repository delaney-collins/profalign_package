#!/usr/bin/env python3
"""
test_task_queue.py - Test task queue for pairwise comparisons

Tests:
1. Task creation and serialization
2. Queue generation from pairs
3. Sequential execution on real MSA files
4. Result collection into ScoreStore
"""

import tempfile
from pathlib import Path

import pytest


from task_queue import (
    PairwiseTask,
    TaskQueue,
    run_single_task,
    run_task_queue_sequential,
    results_to_distance_store,
)
from pipeline_utils import ScoreStore


def test_task_creation():
    """Test task creation and serialization."""
    print("\n" + "=" * 60)
    print("Test 1: Task Creation")
    print("=" * 60)
    
    task = PairwiseTask(
        query_profile=Path("/data/P001.fasta"),
        target_profile=Path("/data/P002.fasta"),
        query_name="P001",
        target_name="P002",
        output_hhr=Path("/results/P001_vs_P002.hhr"),
    )
    
    # Test to_dict / from_dict
    d = task.to_dict()
    task2 = PairwiseTask.from_dict(d)
    
    assert task.query_name == task2.query_name
    assert task.target_name == task2.target_name
    assert str(task.output_hhr) == str(task2.output_hhr)
    
    print(f"  ✓ Task serialization works")
    print(f"    {task.query_name} vs {task.target_name} -> {task.output_hhr.name}")
    print("\n  PASSED ✓")


def test_queue_operations():
    """Test queue creation and file I/O."""
    print("\n" + "=" * 60)
    print("Test 2: Queue Operations")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create queue manually
        queue = TaskQueue()
        
        for i in range(3):
            task = PairwiseTask(
                query_profile=tmpdir / f"P00{i}.fasta",
                target_profile=tmpdir / f"P00{i+1}.fasta",
                query_name=f"P00{i}",
                target_name=f"P00{i+1}",
                output_hhr=tmpdir / f"P00{i}_vs_P00{i+1}.hhr",
            )
            queue.add_task(task)
        
        assert len(queue) == 3
        print(f"  ✓ Created queue with {len(queue)} tasks")
        
        # Save to file
        queue_file = tmpdir / "queue.tsv"
        queue.save_to_file(queue_file)
        
        print(f"  ✓ Saved queue to file")
        print(f"    Contents:")
        with open(queue_file) as f:
            for line in f:
                print(f"      {line.rstrip()[:80]}...")
        
        # Load from file
        queue2 = TaskQueue.load_from_file(queue_file)
        assert len(queue2) == 3
        print(f"  ✓ Loaded queue: {len(queue2)} tasks")
        
        # Test completed tracking
        pending = queue2.get_pending_tasks()
        assert len(pending) == 3
        
        queue2.mark_completed("P000", "P001")
        pending = queue2.get_pending_tasks()
        assert len(pending) == 2
        print(f"  ✓ Completed tracking: {len(pending)} pending after 1 done")
    
    print("\n  PASSED ✓")


def test_queue_from_pairs():
    """Test queue generation from profile pairs."""
    print("\n" + "=" * 60)
    print("Test 3: Queue from Pairs")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy profile files
        profile_paths = {}
        for name in ["P001", "P002", "P003", "P004"]:
            path = tmpdir / f"{name}.fasta"
            path.write_text(f">{name}\nACDEFGHIKLMN\n")
            profile_paths[name] = path
        
        # Generate queue from pairs
        pairs = [
            ("P001", "P002"),
            ("P001", "P003"),
            ("P002", "P003"),
            ("P003", "P004"),
        ]
        
        queue = TaskQueue()
        queue.add_tasks_from_pairs(pairs, profile_paths, tmpdir / "results")
        
        assert len(queue) == 4
        print(f"  ✓ Generated {len(queue)} tasks from {len(pairs)} pairs")
        
        for task in queue.tasks:
            print(f"    {task.query_name} vs {task.target_name}")
    
    print("\n  PASSED ✓")


@pytest.mark.integration
def test_execution_on_real_data(msa_dir: Path, has_hhalign):
    """Test actual execution on real MSA files."""
    print("\n" + "=" * 60)
    print("Test 4: Execution on Real Data")
    print("=" * 60)
    
    # Get first 3 MSA files
    msa_files = sorted(msa_dir.glob("*.fasta"))[:3]
    
    if len(msa_files) < 3:
        print(f"  SKIPPED: Need at least 3 MSA files in {msa_dir}")
        return
    
    print(f"  Using {len(msa_files)} MSA files:")
    for f in msa_files:
        print(f"    - {f.name}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create profile paths dict
        profile_paths = {f.stem: f for f in msa_files}
        profile_names = list(profile_paths.keys())
        
        # Generate all pairs
        pairs = []
        for i in range(len(profile_names)):
            for j in range(i + 1, len(profile_names)):
                pairs.append((profile_names[i], profile_names[j]))
        
        print(f"  Generated {len(pairs)} pairs to compare")
        
        # Create queue
        queue = TaskQueue()
        queue.add_tasks_from_pairs(pairs, profile_paths, tmpdir / "hhr")
        
        # Run sequentially
        print(f"  Running comparisons...")
        results = run_task_queue_sequential(
            queue,
            add_consensus=True,
            max_distance=12.0,
        )
        
        # Check results
        successful = [r for r in results if r.success]
        print(f"\n  Results: {len(successful)}/{len(results)} successful")
        
        for r in results:
            status = "✓" if r.success else "✗"
            print(f"    {status} {r.query_name} vs {r.target_name}: "
                  f"dist={r.distance:.3f}, prob={r.probability:.1f}%, "
                  f"cols={r.aligned_cols}")
        
        # Convert to ScoreStore
        store = ScoreStore(tmpdir / "distances.tsv")
        results_to_distance_store(results, store)
        
        print(f"\n  ✓ Saved {len(store.pairs)} distances to file")
        
        # Verify symmetric lookup
        if results:
            r = results[0]
            d1 = store.get_distance(r.query_name, r.target_name)
            d2 = store.get_distance(r.target_name, r.query_name)
            assert d1 == d2
            print(f"  ✓ Symmetric lookup verified")
    
    print("\n  PASSED ✓")


def main():
    print("=" * 60)
    print("Task Queue Tests")
    print("=" * 60)
    
    test_task_creation()
    test_queue_operations()
    test_queue_from_pairs()
    
    # Test with real data if available
    msa_dir = Path("../data/msa")
    if msa_dir.exists():
        test_execution_on_real_data(msa_dir)
    else:
        print(f"\n  Note: {msa_dir} not found, skipping real data test")
        print(f"  Run from profalign/ directory with data available")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
