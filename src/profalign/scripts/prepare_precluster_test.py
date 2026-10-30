#!/usr/bin/env python3
"""
prepare_precluster_test.py - Prepare test dataset for precluster.py validation

Takes MSA files, removes gaps, and creates a single "raw" FASTA file.
Also creates a ground truth mapping (sequence -> original MSA) for validation.

Usage:
    python prepare_precluster_test.py msa_dir/ output_dir/
    
Then:
    python precluster.py output_dir/raw_sequences.fasta output_dir/clusters/
    python prepare_precluster_test.py --validate output_dir/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def read_msa_files(msa_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Read all MSA files from directory.
    
    Returns:
        Dict mapping msa_name -> {seq_id: sequence_with_gaps}
    """
    msa_data = {}
    
    for ext in ['*.fasta', '*.fa', '*.a3m']:
        for fpath in msa_dir.glob(ext):
            msa_name = fpath.stem
            sequences = {}
            
            current_id = None
            current_seq = []
            
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_id:
                            sequences[current_id] = ''.join(current_seq)
                        current_id = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line)
                
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
            
            if sequences:
                msa_data[msa_name] = sequences
                print(f"  {msa_name}: {len(sequences)} sequences")
    
    return msa_data


def remove_gaps(sequence: str) -> str:
    """Remove gap characters from sequence."""
    return sequence.replace('-', '').replace('.', '')


def create_raw_fasta(msa_data: Dict[str, Dict[str, str]], output_path: Path) -> Dict[str, str]:
    """
    Create raw FASTA file from MSA data.
    
    Returns:
        Dict mapping seq_id -> original_msa_name (ground truth)
    """
    ground_truth = {}
    
    with open(output_path, 'w') as f:
        for msa_name, sequences in msa_data.items():
            for seq_id, seq_with_gaps in sequences.items():
                # Remove gaps
                raw_seq = remove_gaps(seq_with_gaps)
                
                # Create unique ID (in case same seq_id in multiple MSAs)
                unique_id = f"{msa_name}__{seq_id}"
                
                # Write to FASTA
                f.write(f">{unique_id}\n")
                # Wrap at 80 chars
                for i in range(0, len(raw_seq), 80):
                    f.write(raw_seq[i:i+80] + '\n')
                
                # Record ground truth
                ground_truth[unique_id] = msa_name
    
    return ground_truth


def save_ground_truth(ground_truth: Dict[str, str], output_path: Path):
    """Save ground truth mapping to TSV file."""
    with open(output_path, 'w') as f:
        f.write("# seq_id\toriginal_msa\n")
        for seq_id, msa_name in sorted(ground_truth.items()):
            f.write(f"{seq_id}\t{msa_name}\n")


def load_ground_truth(filepath: Path) -> Dict[str, str]:
    """Load ground truth mapping from TSV file."""
    ground_truth = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                ground_truth[parts[0]] = parts[1]
    return ground_truth


def load_clusters(clusters_tsv: Path) -> Dict[str, List[str]]:
    """
    Load MMseqs2 clusters from TSV file.
    
    Returns:
        Dict mapping representative_id -> list of member_ids
    """
    clusters = {}
    with open(clusters_tsv) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep_id, member_id = parts[0], parts[1]
                if rep_id not in clusters:
                    clusters[rep_id] = []
                clusters[rep_id].append(member_id)
    return clusters


def validate_clustering(ground_truth: Dict[str, str], clusters: Dict[str, List[str]]) -> dict:
    """
    Compare clustering results with ground truth.
    
    Returns statistics about how well clusters match original MSAs.
    """
    stats = {
        'total_sequences': len(ground_truth),
        'total_clusters': len(clusters),
        'total_original_msas': len(set(ground_truth.values())),
        'perfect_clusters': 0,      # Clusters that exactly match one MSA
        'pure_clusters': 0,         # Clusters where all seqs from same MSA
        'mixed_clusters': 0,        # Clusters with seqs from multiple MSAs
        'split_msas': 0,            # MSAs split across multiple clusters
        'cluster_details': [],
    }
    
    # Analyze each cluster
    for rep_id, members in clusters.items():
        # Get original MSAs for all members
        msa_counts = defaultdict(int)
        for member in members:
            if member in ground_truth:
                msa_counts[ground_truth[member]] += 1
        
        cluster_size = len(members)
        num_msas = len(msa_counts)
        dominant_msa = max(msa_counts.keys(), key=lambda x: msa_counts[x]) if msa_counts else None
        dominant_count = msa_counts[dominant_msa] if dominant_msa else 0
        
        # Classify cluster
        if num_msas == 1:
            stats['pure_clusters'] += 1
            # Check if it's the complete MSA
            original_msa_size = sum(1 for v in ground_truth.values() if v == dominant_msa)
            if cluster_size == original_msa_size:
                stats['perfect_clusters'] += 1
        else:
            stats['mixed_clusters'] += 1
        
        stats['cluster_details'].append({
            'representative': rep_id,
            'size': cluster_size,
            'num_source_msas': num_msas,
            'dominant_msa': dominant_msa,
            'purity': dominant_count / cluster_size if cluster_size > 0 else 0,
            'msa_breakdown': dict(msa_counts),
        })
    
    # Check how many MSAs were split
    msa_to_clusters = defaultdict(set)
    for rep_id, members in clusters.items():
        for member in members:
            if member in ground_truth:
                msa_to_clusters[ground_truth[member]].add(rep_id)
    
    stats['split_msas'] = sum(1 for clusters_set in msa_to_clusters.values() if len(clusters_set) > 1)
    stats['msa_distribution'] = {msa: len(clusters_set) for msa, clusters_set in msa_to_clusters.items()}
    
    return stats


def print_validation_report(stats: dict):
    """Print human-readable validation report."""
    print("\n" + "=" * 60)
    print("PRE-CLUSTERING VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\nOverview:")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Original MSAs: {stats['total_original_msas']}")
    print(f"  Clusters found: {stats['total_clusters']}")
    
    print(f"\nCluster Quality:")
    print(f"  Perfect clusters (exact MSA match): {stats['perfect_clusters']}")
    print(f"  Pure clusters (single MSA source): {stats['pure_clusters']}")
    print(f"  Mixed clusters (multiple MSAs): {stats['mixed_clusters']}")
    
    print(f"\nMSA Integrity:")
    print(f"  MSAs kept intact: {stats['total_original_msas'] - stats['split_msas']}")
    print(f"  MSAs split across clusters: {stats['split_msas']}")
    
    if stats['split_msas'] > 0:
        print(f"\n  Split MSAs detail:")
        for msa, num_clusters in stats['msa_distribution'].items():
            if num_clusters > 1:
                print(f"    {msa}: split into {num_clusters} clusters")
    
    # Show mixed clusters if any
    mixed = [c for c in stats['cluster_details'] if c['num_source_msas'] > 1]
    if mixed:
        print(f"\nMixed Clusters Detail (top 5):")
        for c in sorted(mixed, key=lambda x: -x['size'])[:5]:
            print(f"  Cluster (size={c['size']}, purity={c['purity']:.1%}):")
            for msa, count in sorted(c['msa_breakdown'].items(), key=lambda x: -x[1]):
                print(f"    {msa}: {count} sequences")
    
    # Overall score
    purity = stats['pure_clusters'] / stats['total_clusters'] if stats['total_clusters'] > 0 else 0
    integrity = (stats['total_original_msas'] - stats['split_msas']) / stats['total_original_msas'] if stats['total_original_msas'] > 0 else 0
    
    print(f"\nOverall Scores:")
    print(f"  Cluster Purity: {purity:.1%} ({stats['pure_clusters']}/{stats['total_clusters']} pure)")
    print(f"  MSA Integrity: {integrity:.1%} ({stats['total_original_msas'] - stats['split_msas']}/{stats['total_original_msas']} intact)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare test dataset for precluster.py validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Prepare raw sequences from MSAs
  python prepare_precluster_test.py ../data/msa ./precluster_test/

  # Step 2: Run pre-clustering
  python precluster.py ./precluster_test/raw_sequences.fasta ./precluster_test/clusters/

  # Step 3: Validate results
  python prepare_precluster_test.py --validate ./precluster_test/
"""
    )
    
    parser.add_argument('input_dir', type=Path, nargs='?',
                        help='MSA directory (for preparation) or test directory (for validation)')
    parser.add_argument('output_dir', type=Path, nargs='?',
                        help='Output directory (for preparation mode)')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation mode (input_dir should be test directory)')
    
    args = parser.parse_args()
    
    if args.validate:
        # Validation mode
        if not args.input_dir:
            parser.error("--validate requires input_dir (test directory)")
        
        test_dir = args.input_dir
        ground_truth_file = test_dir / 'ground_truth.tsv'
        clusters_file = test_dir / 'clusters' / 'clusters.tsv'
        
        if not ground_truth_file.exists():
            print(f"Error: Ground truth file not found: {ground_truth_file}")
            sys.exit(1)
        
        if not clusters_file.exists():
            print(f"Error: Clusters file not found: {clusters_file}")
            print("Run precluster.py first!")
            sys.exit(1)
        
        print("Loading ground truth...")
        ground_truth = load_ground_truth(ground_truth_file)
        print(f"  {len(ground_truth)} sequences")
        
        print("Loading clusters...")
        clusters = load_clusters(clusters_file)
        print(f"  {len(clusters)} clusters")
        
        print("Validating...")
        stats = validate_clustering(ground_truth, clusters)
        print_validation_report(stats)
        
    else:
        # Preparation mode
        if not args.input_dir or not args.output_dir:
            parser.error("Preparation mode requires input_dir and output_dir")
        
        msa_dir = args.input_dir
        output_dir = args.output_dir
        
        if not msa_dir.exists():
            print(f"Error: MSA directory not found: {msa_dir}")
            sys.exit(1)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Reading MSA files from {msa_dir}...")
        msa_data = read_msa_files(msa_dir)
        
        if not msa_data:
            print("Error: No MSA files found!")
            sys.exit(1)
        
        total_seqs = sum(len(seqs) for seqs in msa_data.values())
        print(f"\nTotal: {len(msa_data)} MSAs, {total_seqs} sequences")
        
        raw_fasta_path = output_dir / 'raw_sequences.fasta'
        print(f"\nCreating raw FASTA: {raw_fasta_path}")
        ground_truth = create_raw_fasta(msa_data, raw_fasta_path)
        
        ground_truth_path = output_dir / 'ground_truth.tsv'
        print(f"Saving ground truth: {ground_truth_path}")
        save_ground_truth(ground_truth, ground_truth_path)
        
        print(f"\nDone! Next steps:")
        print(f"  1. python precluster.py {raw_fasta_path} {output_dir}/clusters/ --identity 0.5")
        print(f"  2. python prepare_precluster_test.py --validate {output_dir}/")


if __name__ == '__main__':
    main()
