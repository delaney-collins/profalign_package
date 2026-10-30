#!/usr/bin/env python3
"""
diagnose_merge.py - Diagnose MSA merge quality

Checks:
1. hhalign output (score, probability, coverage)
2. Alignment table from -atab
3. Visual comparison of alignments
4. Comparison with/without consensus
"""

import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))

from formats import read_fasta, write_fasta
from merge import add_consensus_to_alignment, compute_column_stats

def run_hhalign_diagnostic(msa1_path, msa2_path, label, tmpdir):
    """Run hhalign and print diagnostics."""
    
    hhr_file = tmpdir / f"{label}_align.hhr"
    atab_file = tmpdir / f"{label}_align.atab"
    
    cmd = [
        'hhalign',
        '-i', str(msa1_path),
        '-t', str(msa2_path),
        '-o', str(hhr_file),
        '-atab', str(atab_file),
        '-id', '100',
        '-diff', 'inf',
        '-M', 'first',
        '-glob',
        '-v', '0'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse results
    prob = 0
    aligned_cols = 0
    query_range = ""
    template_range = ""
    
    if hhr_file.exists():
        with open(hhr_file) as f:
            for line in f:
                if line.startswith('Probab='):
                    parts = line.split()
                    for p in parts:
                        if p.startswith('Probab='):
                            prob = float(p.split('=')[1])
                        if p.startswith('Aligned_cols='):
                            aligned_cols = int(p.split('=')[1])
                if line.strip().startswith('1 '):
                    # Hit line
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if '-' in p and p[0].isdigit():
                            if not query_range:
                                query_range = p
                            else:
                                template_range = p
                                break
    
    # Count atab positions
    valid_positions = 0
    if atab_file.exists():
        with open(atab_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        p1, p2 = int(parts[0]), int(parts[1])
                        if p1 > 0 and p2 > 0:
                            valid_positions += 1
                    except:
                        pass
    
    return {
        'prob': prob,
        'aligned_cols': aligned_cols,
        'query_range': query_range,
        'template_range': template_range,
        'atab_positions': valid_positions
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python diagnose_merge.py <msa1.fasta> <msa2.fasta>")
        sys.exit(1)
    
    msa1_path = Path(sys.argv[1])
    msa2_path = Path(sys.argv[2])
    
    ali1 = read_fasta(msa1_path)
    ali2 = read_fasta(msa2_path)
    
    print("=" * 70)
    print("MSA Merge Diagnostics")
    print("=" * 70)
    
    print(f"\nMSA1: {msa1_path.name} ({len(ali1)} seqs, {ali1.get_length()} cols)")
    print(f"MSA2: {msa2_path.name} ({len(ali2)} seqs, {ali2.get_length()} cols)")
    
    # Show first sequence info
    first1 = ali1.sequences[0]
    first2 = ali2.sequences[0]
    ungapped1 = len(first1.sequence.replace('-', ''))
    ungapped2 = len(first2.sequence.replace('-', ''))
    
    print(f"\nFirst sequence in MSA1: {first1.id}")
    print(f"  Length: {ungapped1} aa (without gaps)")
    
    print(f"\nFirst sequence in MSA2: {first2.id}")
    print(f"  Length: {ungapped2} aa (without gaps)")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test 1: Original MSAs (no consensus)
        print("\n" + "-" * 70)
        print("Test 1: hhalign on ORIGINAL MSAs (no consensus)")
        print("-" * 70)
        
        result1 = run_hhalign_diagnostic(msa1_path, msa2_path, "original", tmpdir)
        
        print(f"  Probability: {result1['prob']:.2f}%")
        print(f"  Aligned columns: {result1['aligned_cols']}")
        print(f"  ATAB positions: {result1['atab_positions']}")
        print(f"  Query range: {result1['query_range']}")
        print(f"  Template range: {result1['template_range']}")
        
        # Test 2: MSAs with consensus added
        print("\n" + "-" * 70)
        print("Test 2: hhalign on MSAs WITH CONSENSUS as first sequence")
        print("-" * 70)
        
        # Add consensus
        ali1_with_cons = add_consensus_to_alignment(ali1, f"{msa1_path.stem}@_cons")
        ali2_with_cons = add_consensus_to_alignment(ali2, f"{msa2_path.stem}@_cons")
        
        # Show consensus info
        cons1 = ali1_with_cons.sequences[0].sequence
        cons2 = ali2_with_cons.sequences[0].sequence
        cons1_ungapped = len(cons1.replace('-', '').replace('x', ''))
        cons2_ungapped = len(cons2.replace('-', '').replace('x', ''))
        
        print(f"\n  Consensus1 length: {cons1_ungapped} confident aa (of {len(cons1)} total)")
        print(f"  Consensus2 length: {cons2_ungapped} confident aa (of {len(cons2)} total)")
        
        # Write and test
        cons1_path = tmpdir / "msa1_with_cons.fasta"
        cons2_path = tmpdir / "msa2_with_cons.fasta"
        write_fasta(ali1_with_cons, cons1_path)
        write_fasta(ali2_with_cons, cons2_path)
        
        result2 = run_hhalign_diagnostic(cons1_path, cons2_path, "with_cons", tmpdir)
        
        print(f"\n  Probability: {result2['prob']:.2f}%")
        print(f"  Aligned columns: {result2['aligned_cols']}")
        print(f"  ATAB positions: {result2['atab_positions']}")
        print(f"  Query range: {result2['query_range']}")
        print(f"  Template range: {result2['template_range']}")
        
        # Comparison
        print("\n" + "-" * 70)
        print("COMPARISON")
        print("-" * 70)
        
        if result2['atab_positions'] > result1['atab_positions']:
            improvement = result2['atab_positions'] / max(result1['atab_positions'], 1)
            print(f"  ✓ With consensus: {improvement:.1f}x more aligned positions!")
        else:
            print(f"  Same or worse with consensus")
        
        print(f"\n  Without consensus: {result1['atab_positions']} positions, {result1['prob']:.1f}% prob")
        print(f"  With consensus:    {result2['atab_positions']} positions, {result2['prob']:.1f}% prob")


if __name__ == "__main__":
    main()
