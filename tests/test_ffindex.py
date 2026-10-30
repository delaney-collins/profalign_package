#!/usr/bin/env python3
"""
test_ffindex.py - Test ffindex-based all-vs-all search

This script tests:
1. Building ffindex database from MSA files
2. Running hhsearch against the database
3. Parsing results into distance matrix
4. Comparing speed with pairwise hhalign

Usage:
    python test_ffindex.py <msa_dir> [num_files]
    
Example:
    python test_ffindex.py ../data/msa 10
"""

import sys
import time
import tempfile
import logging
from pathlib import Path


from formats import read_fasta, get_alignment_stats
from ffindex_db import (
    check_ffindex_tools, 
    get_hhlib_path,
    run_all_vs_all_hhsearch
)
from hhr_parser import parse_multiple_hhr, build_distance_matrix, apply_name_mapping
from clustering import build_tree_from_distance_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ffindex.py <msa_dir> [num_files]")
        print("Example: python test_ffindex.py ../data/msa 10")
        sys.exit(1)
    
    msa_dir = Path(sys.argv[1])
    num_files = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not msa_dir.exists():
        print(f"ERROR: Directory not found: {msa_dir}")
        sys.exit(1)
    
    msa_files = sorted(msa_dir.glob("*.fasta"))[:num_files]
    
    if len(msa_files) < 2:
        print(f"ERROR: Need at least 2 MSA files, found {len(msa_files)}")
        sys.exit(1)
    
    print("=" * 70)
    print("FFindex All-vs-All Search Test")
    print("=" * 70)
    
    # Check prerequisites
    print("\n[1] Checking prerequisites...")
    
    tools = check_ffindex_tools()
    ffindex_ok = tools.get('ffindex_build') is not None
    
    if not ffindex_ok:
        print("  ERROR: ffindex_build not found!")
        print("  Install with: conda install -c bioconda ffindex")
        sys.exit(1)
    print(f"  ffindex_build: ✓")
    
    hhlib = get_hhlib_path()
    if hhlib:
        print(f"  HHLIB: ✓ ({hhlib})")
    else:
        print("  HHLIB: ✗ (cs219 prefilter not available)")
    
    # Show input files
    print(f"\n[2] Input: {len(msa_files)} MSA files")
    for f in msa_files[:5]:
        ali = read_fasta(f)
        stats = get_alignment_stats(ali)
        print(f"  {f.name}: {stats['num_sequences']} seqs, {stats['alignment_length']} cols")
    if len(msa_files) > 5:
        print(f"  ... and {len(msa_files) - 5} more")
    
    # Run all-vs-all with ffindex
    print("\n[3] Running all-vs-all hhsearch (ffindex)...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        start_time = time.time()
        
        result = run_all_vs_all_hhsearch(
            msa_files,
            work_dir=tmpdir,
            use_cs219=(hhlib is not None),
            add_consensus=False,  # Don't add consensus - it breaks match columns
            cpu=1
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n  Completed in {elapsed:.1f} seconds")
        print(f"  {len(result['hhr_files'])} searches performed")
        print(f"  Speed: {len(msa_files) / elapsed:.1f} profiles/second")
        
        # Parse results
        print("\n[4] Parsing results...")
        
        hhr_results = parse_multiple_hhr(result['hhr_files'])
        
        # Apply name mapping (convert first_seq_id -> profile_name)
        if 'name_mapping' in result:
            hhr_results = apply_name_mapping(hhr_results, result['name_mapping'])
        
        distance_matrix, names = build_distance_matrix(
            hhr_results,
            normalize=True,
            symmetrize='max',
            distance_type='log'
        )
        
        print(f"  Matrix size: {distance_matrix.shape[0]} x {distance_matrix.shape[1]}")
        
        # Build tree
        print("\n[5] Building UPGMA tree...")
        
        newick = build_tree_from_distance_matrix(distance_matrix, names, method='upgma')
        
        print(f"  Newick: {newick[:100]}..." if len(newick) > 100 else f"  Newick: {newick}")
        
        # Show distance matrix (truncated)
        print("\n[6] Distance matrix (first 5x5):")
        n_show = min(5, len(names))
        print("    " + " " * 12 + "  ".join(f"{n[:8]:>8}" for n in names[:n_show]))
        for i in range(n_show):
            row = "  ".join(f"{distance_matrix[i,j]:8.3f}" for j in range(n_show))
            print(f"    {names[i][:12]:<12} {row}")
    
    print("\n" + "=" * 70)
    print("SUCCESS: FFindex test completed!")
    print("=" * 70)
    
    # Performance estimate
    print(f"\nPerformance estimate for 30 profiles:")
    print(f"  FFindex (n searches): ~{30 / (len(msa_files) / elapsed):.0f} seconds")
    print(f"  Pairwise (n² comparisons): ~{30*30 * 0.25:.0f} seconds")


if __name__ == "__main__":
    main()
