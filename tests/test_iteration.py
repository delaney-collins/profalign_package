#!/usr/bin/env python3
"""
Test iteration with synthetic distance data.

This tests the Yuri's algorithm implementation without requiring hhalign.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add lib to path

from pipeline_utils import ScoreStore, MergeHistory
from clustering import hierarchical_clustering, linkage_to_newick
from guided_align import (
    scipy_node_to_tree_node,
    extract_subtrees_at_depth,
    get_nontrivial_subtrees,
    calc_tree_weighted_depth,
    print_subtree
)
from scipy.cluster.hierarchy import to_tree
import numpy as np


def create_synthetic_distances(profiles: list, families: dict) -> dict:
    """
    Create synthetic distance data.
    
    Args:
        profiles: List of profile names
        families: Dict mapping family name -> list of profile names
        
    Returns:
        Dict of (p1, p2) -> distance
    """
    distances = {}
    
    # Build family lookup
    profile_to_family = {}
    for family, members in families.items():
        for m in members:
            profile_to_family[m] = family
    
    for i, p1 in enumerate(profiles):
        for p2 in profiles[i+1:]:
            fam1 = profile_to_family.get(p1, "unknown")
            fam2 = profile_to_family.get(p2, "unknown")
            
            if fam1 == fam2:
                # Same family - low distance
                dist = 0.1 + np.random.uniform(0, 0.4)
            else:
                # Different family - high distance
                dist = 5.0 + np.random.uniform(0, 3.0)
            
            distances[(p1, p2)] = dist
    
    return distances


def main():
    print("=" * 60)
    print("TEST: iteration with synthetic data")
    print("=" * 60)
    
    # Define test profiles and families
    families = {
        'Cas10': ['Cas10_1', 'Cas10_2', 'Cas10_3', 'Cas10_4'],
        'Cas12': ['Cas12_1', 'Cas12_2', 'Cas12_3'],
        'Cam': ['Cam_1', 'Cam_2'],
    }
    
    profiles = []
    for members in families.values():
        profiles.extend(members)
    
    print(f"\nProfiles: {profiles}")
    print(f"Families: {families}")
    
    # Create synthetic distances
    np.random.seed(42)
    distances = create_synthetic_distances(profiles, families)
    
    print(f"\nSynthetic distances: {len(distances)} pairs")
    
    # Build distance matrix
    names = sorted(profiles)
    n = len(names)
    matrix = np.full((n, n), 10.0)
    np.fill_diagonal(matrix, 0)
    
    name_to_idx = {name: i for i, name in enumerate(names)}
    
    for (p1, p2), dist in distances.items():
        i, j = name_to_idx[p1], name_to_idx[p2]
        matrix[i, j] = dist
        matrix[j, i] = dist
    
    print("\nDistance matrix:")
    print("        ", end="")
    for name in names:
        print(f"{name[:7]:>8}", end="")
    print()
    for i, name in enumerate(names):
        print(f"{name[:7]:>8}", end="")
        for j in range(n):
            print(f"{matrix[i,j]:8.2f}", end="")
        print()
    
    # Build UPGMA tree
    print("\n" + "=" * 60)
    print("UPGMA TREE")
    print("=" * 60)
    
    linkage_matrix = hierarchical_clustering(matrix, method='upgma')
    newick = linkage_to_newick(linkage_matrix, names)
    print(f"Newick: {newick}")
    
    # Convert to TreeNode
    scipy_tree = to_tree(linkage_matrix)
    tree = scipy_node_to_tree_node(scipy_tree, names)
    calc_tree_weighted_depth(tree, weight_mode=2)  # Yuri's default: unweighted
    
    print("\nTree structure:")
    print_subtree(tree)
    
    # Extract subtrees at different depths
    print("\n" + "=" * 60)
    print("SUBTREE EXTRACTION (tree_tips)")
    print("=" * 60)
    
    for depth in [0.5, 1.0, 1.2, 2.0]:
        subtrees = extract_subtrees_at_depth(tree, depth)
        nontrivial = get_nontrivial_subtrees(subtrees)
        
        print(f"\nDepth threshold: {depth}")
        print(f"  Total subtrees: {len(subtrees)}")
        print(f"  Non-trivial: {len(nontrivial)}")
        
        for i, st in enumerate(subtrees, 1):
            leaves = st.get_leaves()
            if st.is_leaf():
                print(f"    Subtree {i}: [singleton] {leaves[0]}")
            else:
                print(f"    Subtree {i}: [{len(leaves)} profiles, h={st.height:.3f}] {leaves}")
    
    # Test with depth=1.2 (Yuri's default)
    print("\n" + "=" * 60)
    print("GUIDED ALIGNMENT SIMULATION (depth=1.2)")
    print("=" * 60)
    
    subtrees = extract_subtrees_at_depth(tree, 1.2)
    
    print(f"\nSubtrees to process: {len(subtrees)}")
    
    for i, st in enumerate(subtrees, 1):
        leaves = st.get_leaves()
        
        if st.is_leaf():
            print(f"\nSubtree {i}: SINGLETON ({leaves[0]}) - no merge needed")
        else:
            print(f"\nSubtree {i}: MERGE {len(leaves)} profiles")
            print(f"  Profiles: {leaves}")
            print(f"  Height: {st.height:.3f}")
            
            # Simulate guided merge order
            def print_merge_order(node, level=0):
                if node.is_leaf():
                    return node.name
                
                prefix = "  " * (level + 1)
                left_result = print_merge_order(node.left, level + 1)
                right_result = print_merge_order(node.right, level + 1)
                
                merge_name = f"merge_{level}"
                print(f"{prefix}MERGE: {left_result} + {right_result} -> {merge_name}")
                
                return merge_name
            
            print("  Merge order:")
            print_merge_order(st)
    
    # Expected results
    print("\n" + "=" * 60)
    print("EXPECTED RESULTS")
    print("=" * 60)
    print("""
At depth=1.2, we expect 3 subtrees (one per family):
1. Cas10 subtree (4 profiles) -> 1 merged profile
   - Merges: (Cas10_1 + Cas10_2) + (Cas10_3 + Cas10_4) 
2. Cas12 subtree (3 profiles) -> 1 merged profile
   - Merges: Cas12_3 + (Cas12_1 + Cas12_2)
3. Cam subtree (2 profiles) -> 1 merged profile
   - Merges: Cam_1 + Cam_2

Result: 9 input profiles -> 3 output profiles
""")
    
    # Validation
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    subtrees = extract_subtrees_at_depth(tree, 1.2)
    nontrivial = get_nontrivial_subtrees(subtrees)
    
    # Check that we get 3 subtrees
    check1 = len(subtrees) == 3
    print(f"✓ 3 subtrees at depth 1.2: {check1} (got {len(subtrees)})")
    
    # Check that all are non-trivial
    check2 = len(nontrivial) == 3
    print(f"✓ All 3 are non-trivial: {check2} (got {len(nontrivial)})")
    
    # Check that each subtree contains exactly one family
    check3 = True
    for st in subtrees:
        leaves = st.get_leaves()
        # All leaves should be from same family
        family = None
        for fam, members in families.items():
            if leaves[0] in members:
                family = fam
                break
        if family:
            all_same = all(leaf in families[family] for leaf in leaves)
            if not all_same:
                check3 = False
                break
    print(f"✓ Each subtree contains one family: {check3}")
    
    if check1 and check2 and check3:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
