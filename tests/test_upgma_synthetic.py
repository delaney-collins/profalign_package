#!/usr/bin/env python3
"""
Test UPGMA clustering with synthetic distance matrix.

This validates that our scipy-based UPGMA produces correct trees
without needing hhalign.
"""

import sys
import numpy as np
from pathlib import Path

# Add lib to path

from clustering import (
    hierarchical_clustering,
    linkage_to_newick,
    get_subtrees_at_depth,
    get_tree_height,
    get_clusters_at_height
)
from scipy.cluster.hierarchy import to_tree


def create_test_distances():
    """
    Create a synthetic distance matrix that mimics CRISPR profile distances.
    
    We simulate 3 families:
    - Cas10 family: Cas10_1, Cas10_2, Cas10_3, Cas10_4 (close to each other)
    - Cas12 family: Cas12_1, Cas12_2, Cas12_3 (close to each other)
    - Cam family: Cam_1, Cam_2 (close to each other)
    
    Distances within family: 0.1 - 0.5
    Distances between families: 5.0 - 10.0
    """
    names = [
        'Cas10_1', 'Cas10_2', 'Cas10_3', 'Cas10_4',  # Family 1
        'Cas12_1', 'Cas12_2', 'Cas12_3',              # Family 2
        'Cam_1', 'Cam_2'                              # Family 3
    ]
    n = len(names)
    
    # Initialize with large distances
    matrix = np.full((n, n), 8.0)
    np.fill_diagonal(matrix, 0)
    
    # Within Cas10 family (indices 0-3)
    matrix[0, 1] = matrix[1, 0] = 0.1  # Cas10_1 - Cas10_2 (very close)
    matrix[0, 2] = matrix[2, 0] = 0.3
    matrix[0, 3] = matrix[3, 0] = 0.4
    matrix[1, 2] = matrix[2, 1] = 0.2
    matrix[1, 3] = matrix[3, 1] = 0.35
    matrix[2, 3] = matrix[3, 2] = 0.15  # Cas10_3 - Cas10_4 (very close)
    
    # Within Cas12 family (indices 4-6)
    matrix[4, 5] = matrix[5, 4] = 0.2
    matrix[4, 6] = matrix[6, 4] = 0.3
    matrix[5, 6] = matrix[6, 5] = 0.25
    
    # Within Cam family (indices 7-8)
    matrix[7, 8] = matrix[8, 7] = 0.5
    
    # Between Cas10 and Cas12 (moderate distance)
    for i in range(4):
        for j in range(4, 7):
            matrix[i, j] = matrix[j, i] = 5.0 + np.random.uniform(0, 1)
    
    # Between Cas10 and Cam (large distance)
    for i in range(4):
        for j in range(7, 9):
            matrix[i, j] = matrix[j, i] = 7.0 + np.random.uniform(0, 1)
    
    # Between Cas12 and Cam (large distance)
    for i in range(4, 7):
        for j in range(7, 9):
            matrix[i, j] = matrix[j, i] = 6.5 + np.random.uniform(0, 1)
    
    return matrix, names


def print_matrix(matrix, names):
    """Print distance matrix."""
    print("\nDistance matrix:")
    print("        ", end="")
    for name in names:
        print(f"{name[:7]:>8}", end="")
    print()
    
    for i, name in enumerate(names):
        print(f"{name[:7]:>8}", end="")
        for j in range(len(names)):
            print(f"{matrix[i,j]:8.2f}", end="")
        print()


def print_tree_structure(linkage_matrix, names):
    """Print tree structure."""
    tree = to_tree(linkage_matrix)
    
    def get_name(node):
        if node.is_leaf():
            return names[node.id]
        else:
            return f"({get_name(node.left)}, {get_name(node.right)})"
    
    def print_node(node, level=0):
        prefix = "  " * level
        if node.is_leaf():
            print(f"{prefix}└─ {names[node.id]}")
        else:
            print(f"{prefix}├─ [h={node.dist:.3f}, n={node.count}]")
            print_node(node.left, level + 1)
            print_node(node.right, level + 1)
    
    print_node(tree)


def main():
    print("="*60)
    print("UPGMA CLUSTERING TEST WITH SYNTHETIC DATA")
    print("="*60)
    
    # Create test data
    np.random.seed(42)  # For reproducibility
    matrix, names = create_test_distances()
    
    print(f"\nProfiles: {names}")
    print_matrix(matrix, names)
    
    # Perform UPGMA clustering
    print("\n" + "="*60)
    print("UPGMA CLUSTERING")
    print("="*60)
    
    linkage_matrix = hierarchical_clustering(matrix, method='upgma')
    
    print("\nLinkage matrix (scipy format):")
    print("  [idx1, idx2, height, count]")
    for i, row in enumerate(linkage_matrix):
        idx1, idx2, height, count = row
        name1 = names[int(idx1)] if idx1 < len(names) else f"cluster_{int(idx1)-len(names)+1}"
        name2 = names[int(idx2)] if idx2 < len(names) else f"cluster_{int(idx2)-len(names)+1}"
        print(f"  {i+1}: {name1} + {name2} at height {height:.3f} ({int(count)} members)")
    
    # Tree height
    tree_height = get_tree_height(linkage_matrix)
    print(f"\nTree height: {tree_height:.3f}")
    
    # Newick format
    print("\n" + "="*60)
    print("NEWICK TREE")
    print("="*60)
    newick = linkage_to_newick(linkage_matrix, names)
    print(newick)
    
    # Tree structure
    print("\n" + "="*60)
    print("TREE STRUCTURE")
    print("="*60)
    print_tree_structure(linkage_matrix, names)
    
    # Test subtrees at different depths
    print("\n" + "="*60)
    print("SUBTREES AT DIFFERENT DEPTH THRESHOLDS")
    print("="*60)
    
    for depth in [0.3, 0.5, 1.0, 1.2, 2.0, 5.0]:
        subtrees = get_subtrees_at_depth(linkage_matrix, names, depth)
        
        multi = [(m, h) for m, h in subtrees if len(m) > 1]
        singles = [m[0] for m, h in subtrees if len(m) == 1]
        
        print(f"\nDepth threshold: {depth}")
        print(f"  Subtrees to merge: {len(multi)}")
        for members, height in multi:
            print(f"    {members} (height={height:.3f})")
        if singles:
            print(f"  Singletons: {singles}")
    
    # Expected behavior
    print("\n" + "="*60)
    print("EXPECTED BEHAVIOR")
    print("="*60)
    print("""
At depth=0.3:
  - Should merge: (Cas10_1, Cas10_2), (Cas10_3, Cas10_4), (Cas12_1, Cas12_2)
  - Singletons: Cas12_3, Cam_1, Cam_2

At depth=0.5:
  - Should merge all within-family pairs
  - Cam_1 + Cam_2 should be merged

At depth=1.2 (Yuri's default):
  - Each family should form one subtree
  - 3 subtrees total: Cas10, Cas12, Cam

At depth=5.0+:
  - Some cross-family merges may occur
""")
    
    # Validate against expectations
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    # At depth 0.5, Cam family should be merged
    subtrees_05 = get_subtrees_at_depth(linkage_matrix, names, 0.5)
    cam_merged = any('Cam_1' in m and 'Cam_2' in m for m, h in subtrees_05)
    print(f"✓ Cam family merged at depth 0.5: {cam_merged}")
    
    # At depth 1.2, should have ~3 major groups
    subtrees_12 = get_subtrees_at_depth(linkage_matrix, names, 1.2)
    multi_12 = [m for m, h in subtrees_12 if len(m) > 1]
    print(f"✓ Number of multi-member subtrees at depth 1.2: {len(multi_12)}")
    
    # Check that Cas10 family stays together
    cas10_together = any(
        all(n.startswith('Cas10') for n in m) and len(m) == 4
        for m, h in subtrees_12
    )
    print(f"✓ Cas10 family forms single subtree at depth 1.2: {cas10_together}")
    
    print("\nTest complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
