#!/usr/bin/env python3
"""
Test UPGMA clustering on real distance matrix from pipeline run.

This script:
1. Loads distances.tsv from a pipeline run
2. Builds UPGMA tree using scipy
3. Outputs Newick tree for visual inspection
4. Shows subtrees at different depth thresholds
"""

import sys
import numpy as np
from pathlib import Path

# Add lib to path

from pipeline_utils import ScoreStore
from clustering import (
    hierarchical_clustering,
    linkage_to_newick,
    get_subtrees_at_depth,
    get_tree_height,
    get_clusters_at_height
)
from scipy.cluster.hierarchy import to_tree


def load_distance_matrix(distances: ScoreStore, profile_names: list):
    """
    Convert ScoreStore to numpy matrix.
    
    Args:
        distances: ScoreStore object
        profile_names: Ordered list of profile names
        
    Returns:
        numpy array (n x n) distance matrix
    """
    n = len(profile_names)
    matrix = np.zeros((n, n))
    
    name_to_idx = {name: i for i, name in enumerate(profile_names)}
    
    for (p1, p2), dist_info in distances.distances.items():
        if p1 in name_to_idx and p2 in name_to_idx:
            i, j = name_to_idx[p1], name_to_idx[p2]
            matrix[i, j] = dist_info['distance']
            matrix[j, i] = dist_info['distance']
    
    return matrix


def print_tree_structure(linkage_matrix, names, indent=0):
    """Print tree structure for debugging."""
    tree = to_tree(linkage_matrix)
    
    def print_node(node, level=0):
        prefix = "  " * level
        if node.is_leaf():
            print(f"{prefix}└─ {names[node.id]}")
        else:
            print(f"{prefix}├─ [height={node.dist:.3f}, count={node.count}]")
            if node.left:
                print_node(node.left, level + 1)
            if node.right:
                print_node(node.right, level + 1)
    
    print_node(tree)


def main():
    # Find the most recent results directory
    work_dir = Path("./results/work")
    
    if not work_dir.exists():
        print("No results/work directory found. Run the pipeline first.")
        print("Usage: python test_upgma.py [path_to_distances.tsv]")
        return 1
    
    # Load distances
    distances_file = work_dir / "distances.tsv"
    if len(sys.argv) > 1:
        distances_file = Path(sys.argv[1])
    
    if not distances_file.exists():
        print(f"Distances file not found: {distances_file}")
        return 1
    
    print(f"Loading distances from: {distances_file}")
    distances = ScoreStore(distances_file)
    distances.load()
    
    # Get unique profile names
    profiles = set()
    for p1, p2 in distances.distances.keys():
        profiles.add(p1)
        profiles.add(p2)
    
    profile_names = sorted(profiles)
    n = len(profile_names)
    
    print(f"Found {n} profiles")
    print(f"Profiles: {profile_names[:10]}..." if n > 10 else f"Profiles: {profile_names}")
    
    # Build distance matrix
    print("\nBuilding distance matrix...")
    matrix = load_distance_matrix(distances, profile_names)
    
    # Check for missing distances (set to high value)
    max_dist = np.max(matrix[matrix < float('inf')])
    default_dist = max_dist * 2  # Use 2x max as default for missing
    
    missing_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i, j] == 0 and i != j:
                matrix[i, j] = default_dist
                matrix[j, i] = default_dist
                missing_count += 1
    
    if missing_count > 0:
        print(f"Warning: {missing_count} missing distances set to {default_dist:.3f}")
    
    print(f"Distance range: {np.min(matrix[matrix > 0]):.3f} - {np.max(matrix):.3f}")
    
    # Perform UPGMA clustering
    print("\nPerforming UPGMA clustering...")
    linkage_matrix = hierarchical_clustering(matrix, method='upgma')
    
    # Get tree height
    tree_height = get_tree_height(linkage_matrix)
    print(f"Tree height: {tree_height:.3f}")
    
    # Convert to Newick
    print("\n" + "="*60)
    print("NEWICK TREE:")
    print("="*60)
    newick = linkage_to_newick(linkage_matrix, profile_names)
    print(newick)
    
    # Save Newick to file
    newick_file = work_dir / "upgma_tree.newick"
    with open(newick_file, 'w') as f:
        f.write(newick + '\n')
    print(f"\nSaved to: {newick_file}")
    
    # Show tree structure
    print("\n" + "="*60)
    print("TREE STRUCTURE:")
    print("="*60)
    print_tree_structure(linkage_matrix, profile_names)
    
    # Test different depth thresholds
    print("\n" + "="*60)
    print("SUBTREES AT DIFFERENT DEPTH THRESHOLDS:")
    print("="*60)
    
    for depth in [0.5, 1.0, 1.2, 2.0, 5.0, 10.0]:
        subtrees = get_subtrees_at_depth(linkage_matrix, profile_names, depth)
        
        # Filter to show only non-singleton subtrees
        multi_member = [(members, h) for members, h in subtrees if len(members) > 1]
        singletons = [members[0] for members, h in subtrees if len(members) == 1]
        
        print(f"\nDepth threshold: {depth}")
        print(f"  Subtrees with >1 member: {len(multi_member)}")
        print(f"  Singletons: {len(singletons)}")
        
        for members, height in multi_member:
            print(f"    [{len(members)} members, height={height:.3f}]: {members}")
    
    # Compare with Yuri's default threshold (dthr = 1.2)
    print("\n" + "="*60)
    print("YURI's DEFAULT (depth=1.2):")
    print("="*60)
    subtrees = get_subtrees_at_depth(linkage_matrix, profile_names, 1.2)
    
    print(f"Total subtrees: {len(subtrees)}")
    for i, (members, height) in enumerate(subtrees):
        if len(members) > 1:
            print(f"  Subtree {i+1}: {members} (height={height:.3f})")
        else:
            print(f"  Singleton: {members[0]}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
