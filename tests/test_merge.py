#!/usr/bin/env python3
"""
test_merge.py - Test the MSA merging functionality

This script tests Yuri's algorithm for profile-guided MSA merging:
1. Read two MSA files
2. Filter columns by quality (gaps, homogeneity)
3. Run hhalign on filtered alignments
4. Merge full alignments using alignment table
5. Output merged MSA

Usage:
    python test_merge.py <msa1.fasta> <msa2.fasta> [output.fasta]
    
Example:
    python test_merge.py ../data/msa/C25_Cam2_1.fasta ../data/msa/C25_Cami_1.fasta merged.fasta
"""

import sys
import logging
from pathlib import Path

# Add lib directory to path

from formats import read_fasta, get_alignment_stats, write_fasta
from merge import (
    compute_column_stats, 
    filter_alignment_columns,
    merge_two_msas
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python test_merge.py <msa1.fasta> <msa2.fasta> [output.fasta]")
        print("Example: python test_merge.py ../data/msa/C25_Cam2_1.fasta ../data/msa/C25_Cami_1.fasta")
        sys.exit(1)
    
    msa1_path = Path(sys.argv[1])
    msa2_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not msa1_path.exists():
        print(f"ERROR: File not found: {msa1_path}")
        sys.exit(1)
    if not msa2_path.exists():
        print(f"ERROR: File not found: {msa2_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("MSA Merging Test (Yuri's Algorithm)")
    print("=" * 70)
    
    # Step 1: Read and show input alignments
    print("\n" + "-" * 70)
    print("[1] Input alignments")
    print("-" * 70)
    
    ali1 = read_fasta(msa1_path)
    ali2 = read_fasta(msa2_path)
    
    stats1 = get_alignment_stats(ali1)
    stats2 = get_alignment_stats(ali2)
    
    print(f"\n  MSA1: {msa1_path.name}")
    print(f"    Sequences: {stats1['num_sequences']}")
    print(f"    Alignment length: {stats1['alignment_length']}")
    print(f"    Seq IDs: {', '.join(ali1.get_ids()[:3])}...")
    
    print(f"\n  MSA2: {msa2_path.name}")
    print(f"    Sequences: {stats2['num_sequences']}")
    print(f"    Alignment length: {stats2['alignment_length']}")
    print(f"    Seq IDs: {', '.join(ali2.get_ids()[:3])}...")
    
    # Step 2: Show column statistics
    print("\n" + "-" * 70)
    print("[2] Column statistics")
    print("-" * 70)
    
    col_stats1 = compute_column_stats(ali1)
    col_stats2 = compute_column_stats(ali2)
    
    # Summary of column quality
    good_cols1 = sum(1 for s in col_stats1 if s.gap_fraction <= 0.667 and s.homogeneity >= 0)
    good_cols2 = sum(1 for s in col_stats2 if s.gap_fraction <= 0.667 and s.homogeneity >= 0)
    
    avg_gap1 = sum(s.gap_fraction for s in col_stats1) / len(col_stats1) if col_stats1 else 0
    avg_gap2 = sum(s.gap_fraction for s in col_stats2) / len(col_stats2) if col_stats2 else 0
    
    avg_hom1 = sum(s.homogeneity for s in col_stats1) / len(col_stats1) if col_stats1 else 0
    avg_hom2 = sum(s.homogeneity for s in col_stats2) / len(col_stats2) if col_stats2 else 0
    
    print(f"\n  MSA1:")
    print(f"    Good columns (gap<=0.667): {good_cols1}/{len(col_stats1)}")
    print(f"    Avg gap fraction: {avg_gap1:.3f}")
    print(f"    Avg homogeneity: {avg_hom1:.3f}")
    
    print(f"\n  MSA2:")
    print(f"    Good columns (gap<=0.667): {good_cols2}/{len(col_stats2)}")
    print(f"    Avg gap fraction: {avg_gap2:.3f}")
    print(f"    Avg homogeneity: {avg_hom2:.3f}")
    
    # Step 3: Filter columns
    print("\n" + "-" * 70)
    print("[3] Column filtering")
    print("-" * 70)
    
    filtered1 = filter_alignment_columns(ali1, gap_threshold=0.667, homogeneity_threshold=0.0)
    filtered2 = filter_alignment_columns(ali2, gap_threshold=0.667, homogeneity_threshold=0.0)
    
    print(f"\n  MSA1: {len(filtered1.filtered_positions)}/{stats1['alignment_length']} columns pass filter")
    print(f"  MSA2: {len(filtered2.filtered_positions)}/{stats2['alignment_length']} columns pass filter")
    
    # Step 4: Merge alignments
    print("\n" + "-" * 70)
    print("[4] Merging alignments")
    print("-" * 70)
    
    try:
        merged = merge_two_msas(
            msa1_path,
            msa2_path,
            output_path=output_path,
            gap_threshold=0.667,
            homogeneity_threshold=0.0,
            global_alignment=True
        )
        
        merged_stats = get_alignment_stats(merged)
        
        print(f"\n  Merged alignment:")
        print(f"    Sequences: {merged_stats['num_sequences']} ({stats1['num_sequences']} + {stats2['num_sequences']})")
        print(f"    Alignment length: {merged_stats['alignment_length']}")
        
        if output_path:
            print(f"\n  Saved to: {output_path}")
        
        # Show first few lines of merged alignment
        print("\n  First 3 sequences (first 60 chars):")
        for seq in merged.sequences[:3]:
            seq_preview = seq.sequence[:60]
            if len(seq.sequence) > 60:
                seq_preview += "..."
            print(f"    {seq.id[:20]:<20} {seq_preview}")
        
        if len(merged.sequences) > 3:
            print(f"    ... and {len(merged.sequences) - 3} more sequences")
            
    except Exception as e:
        print(f"\n  ERROR: Merging failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Success
    print("\n" + "=" * 70)
    print("SUCCESS: Merge test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
