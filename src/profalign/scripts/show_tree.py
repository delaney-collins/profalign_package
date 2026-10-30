#!/usr/bin/env python3
"""
show_tree.py - Run all-vs-all and display full tree
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))

from formats import read_fasta
from ffindex_db import run_all_vs_all_hhsearch, get_hhlib_path
from hhr_parser import parse_multiple_hhr, build_distance_matrix, apply_name_mapping
from clustering import build_tree_from_distance_matrix

def main():
    if len(sys.argv) < 2:
        print("Usage: python show_tree.py <msa_dir> [num_files]")
        sys.exit(1)
    
    msa_dir = Path(sys.argv[1])
    num_files = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    
    msa_files = sorted(msa_dir.glob("*.fasta"))[:num_files]
    print(f"Processing {len(msa_files)} MSA files...\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run all-vs-all
        result = run_all_vs_all_hhsearch(
            msa_files,
            work_dir=tmpdir,
            use_cs219=(get_hhlib_path() is not None),
            add_consensus=False,
            cpu=1
        )
        
        # Parse and build matrix
        hhr_results = parse_multiple_hhr(result['hhr_files'])
        if 'name_mapping' in result:
            hhr_results = apply_name_mapping(hhr_results, result['name_mapping'])
        
        distance_matrix, names = build_distance_matrix(
            hhr_results,
            normalize=True,
            symmetrize='max',
            distance_type='log'
        )
        
        # Build tree
        newick = build_tree_from_distance_matrix(distance_matrix, names, method='upgma')
        
        # Print full Newick
        print("=" * 70)
        print("FULL NEWICK TREE:")
        print("=" * 70)
        print(newick)
        print()
        
        # Pretty print tree structure
        print("=" * 70)
        print("TREE STRUCTURE (indented):")
        print("=" * 70)
        print_tree_indented(newick)
        
        # Save to file
        with open("tree_output.nwk", "w") as f:
            f.write(newick)
        print(f"\nTree saved to: tree_output.nwk")


def print_tree_indented(newick: str, indent: int = 0):
    """Pretty print Newick tree with indentation."""
    # Simple parser for visualization
    result = []
    depth = 0
    current = ""
    
    for char in newick:
        if char == '(':
            if current.strip():
                result.append(("  " * depth) + current.strip())
                current = ""
            depth += 1
            result.append(("  " * (depth-1)) + "(")
        elif char == ')':
            if current.strip():
                result.append(("  " * depth) + current.strip())
                current = ""
            depth -= 1
            result.append(("  " * depth) + ")")
        elif char == ',':
            if current.strip():
                result.append(("  " * depth) + current.strip())
                current = ""
        else:
            current += char
    
    if current.strip() and current.strip() != ';':
        result.append(current.strip())
    
    # Group by clades
    print("\nClade summary:")
    leaves = [line.strip() for line in result if ':' in line and '(' not in line and ')' not in line]
    
    # Extract just names
    names = []
    for leaf in leaves:
        name = leaf.split(':')[0].strip()
        if name:
            names.append(name)
    
    # Group by prefix
    groups = {}
    for name in names:
        # Extract family: C25_Cas10d_1 -> Cas10d
        parts = name.replace('C25_', '').split('_')
        if len(parts) >= 1:
            family = parts[0]
            if family not in groups:
                groups[family] = []
            groups[family].append(name)
    
    print("\nProfiles by family:")
    for family in sorted(groups.keys()):
        members = groups[family]
        print(f"  {family}: {len(members)} profiles")
        for m in sorted(members):
            print(f"    - {m}")


if __name__ == "__main__":
    main()
