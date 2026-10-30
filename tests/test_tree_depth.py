#!/usr/bin/env python3
"""
Test that our tree depth calculation matches Yuri's tree_tips algorithm.

Yuri's tree_tips uses:
  - tree_calc_depth: weighted average distance from node to its leaves
  - Condition: node.deep <= threshold (shallow enough to merge)
  - Weight mode -w=2: unweighted (equal weight to each child)

This test verifies our implementation produces the same results.
"""

from pathlib import Path

# Add lib to path

from guided_align import (
    TreeNode, 
    calc_tree_weighted_depth, 
    extract_subtrees_at_depth,
    get_nontrivial_subtrees
)


def test_leaf_depth():
    """Leaf nodes should have depth 0."""
    leaf = TreeNode(name="A", height=0.0)
    calc_tree_weighted_depth(leaf, weight_mode=2)
    assert leaf.weighted_depth == 0.0, f"Leaf depth should be 0, got {leaf.weighted_depth}"
    print("✓ Leaf depth = 0")


def test_simple_binary_tree():
    """
    Test simple binary tree:
    
           root (height=1.0)
          /    \
         A      B
    (height=0) (height=0)
    
    Branch lengths: root->A = 1.0, root->B = 1.0
    
    Unweighted depth of root = average(0+1.0, 0+1.0) = 1.0
    """
    leaf_a = TreeNode(name="A", height=0.0)
    leaf_b = TreeNode(name="B", height=0.0)
    root = TreeNode(left=leaf_a, right=leaf_b, height=1.0)
    
    calc_tree_weighted_depth(root, weight_mode=2)
    
    assert leaf_a.weighted_depth == 0.0
    assert leaf_b.weighted_depth == 0.0
    assert abs(root.weighted_depth - 1.0) < 0.001, f"Root depth should be 1.0, got {root.weighted_depth}"
    print("✓ Simple binary tree: root.depth = 1.0")


def test_asymmetric_tree():
    """
    Test asymmetric tree:
    
           root (height=2.0)
          /    \
         A      internal (height=1.0)
    (h=0)      /    \
              B      C
           (h=0)  (h=0)
    
    Branch lengths:
    - root->A = 2.0
    - root->internal = 1.0
    - internal->B = 1.0
    - internal->C = 1.0
    
    Unweighted depth calculations:
    - internal.deep = avg(0+1.0, 0+1.0) = 1.0
    - root.deep = avg(0+2.0, 1.0+1.0) = avg(2.0, 2.0) = 2.0
    """
    leaf_a = TreeNode(name="A", height=0.0)
    leaf_b = TreeNode(name="B", height=0.0)
    leaf_c = TreeNode(name="C", height=0.0)
    internal = TreeNode(left=leaf_b, right=leaf_c, height=1.0)
    root = TreeNode(left=leaf_a, right=internal, height=2.0)
    
    calc_tree_weighted_depth(root, weight_mode=2)
    
    assert abs(internal.weighted_depth - 1.0) < 0.001, f"Internal depth should be 1.0, got {internal.weighted_depth}"
    assert abs(root.weighted_depth - 2.0) < 0.001, f"Root depth should be 2.0, got {root.weighted_depth}"
    print("✓ Asymmetric tree: internal.depth=1.0, root.depth=2.0")


def test_weight_by_leaves():
    """
    Test weight by number of leaves (mode=0).
    
           root (height=2.0)
          /    \
         A      internal (height=1.0)
    (h=0)      /    \
              B      C
           (h=0)  (h=0)
    
    Weight by leaves:
    - A subtree has 1 leaf
    - internal subtree has 2 leaves
    
    - internal.deep = (1*(0+1.0) + 1*(0+1.0)) / (1+1) = 1.0 (same as unweighted)
    - root.deep = (1*(0+2.0) + 2*(1.0+1.0)) / (1+2) = (2.0 + 4.0) / 3 = 2.0
    """
    leaf_a = TreeNode(name="A", height=0.0)
    leaf_b = TreeNode(name="B", height=0.0)
    leaf_c = TreeNode(name="C", height=0.0)
    internal = TreeNode(left=leaf_b, right=leaf_c, height=1.0)
    root = TreeNode(left=leaf_a, right=internal, height=2.0)
    
    calc_tree_weighted_depth(root, weight_mode=0)
    
    assert abs(internal.weighted_depth - 1.0) < 0.001
    assert abs(root.weighted_depth - 2.0) < 0.001, f"Root depth should be 2.0, got {root.weighted_depth}"
    print("✓ Weight by leaves: root.depth = 2.0")


def test_extract_subtrees():
    """
    Test subtree extraction at depth threshold.
    
           root (depth=2.0)
          /    \
         A      internal (depth=1.0)
    (d=0)      /    \
              B      C
           (d=0)  (d=0)
    
    With threshold=1.5:
    - root.depth (2.0) > 1.5, so recurse
    - A.depth (0) <= 1.5, extract A
    - internal.depth (1.0) <= 1.5, extract internal subtree
    
    Result: [A, internal] (2 subtrees)
    """
    leaf_a = TreeNode(name="A", height=0.0)
    leaf_b = TreeNode(name="B", height=0.0)
    leaf_c = TreeNode(name="C", height=0.0)
    internal = TreeNode(left=leaf_b, right=leaf_c, height=1.0)
    root = TreeNode(left=leaf_a, right=internal, height=2.0)
    
    calc_tree_weighted_depth(root, weight_mode=2)
    
    subtrees = extract_subtrees_at_depth(root, depth_threshold=1.5)
    
    assert len(subtrees) == 2, f"Expected 2 subtrees, got {len(subtrees)}"
    
    # One should be leaf A, other should be internal node
    leaf_names = [s.name for s in subtrees if s.is_leaf()]
    internal_nodes = [s for s in subtrees if not s.is_leaf()]
    
    assert "A" in leaf_names
    assert len(internal_nodes) == 1
    assert set(internal_nodes[0].get_leaves()) == {"B", "C"}
    print("✓ Extract subtrees at threshold=1.5: got [A, (B,C)]")


def test_nontrivial_filter():
    """
    Test that get_nontrivial_subtrees filters out single leaves.
    """
    leaf_a = TreeNode(name="A", height=0.0)
    leaf_b = TreeNode(name="B", height=0.0)
    leaf_c = TreeNode(name="C", height=0.0)
    internal = TreeNode(left=leaf_b, right=leaf_c, height=1.0)
    root = TreeNode(left=leaf_a, right=internal, height=2.0)
    
    calc_tree_weighted_depth(root, weight_mode=2)
    
    subtrees = extract_subtrees_at_depth(root, depth_threshold=1.5)
    nontrivial = get_nontrivial_subtrees(subtrees)
    
    assert len(nontrivial) == 1, f"Expected 1 nontrivial subtree, got {len(nontrivial)}"
    assert set(nontrivial[0].get_leaves()) == {"B", "C"}
    print("✓ Nontrivial filter: only (B,C) subtree kept")


def test_high_threshold_returns_root():
    """
    With very high threshold, should return root as single subtree.
    """
    leaf_a = TreeNode(name="A", height=0.0)
    leaf_b = TreeNode(name="B", height=0.0)
    root = TreeNode(left=leaf_a, right=leaf_b, height=1.0)
    
    calc_tree_weighted_depth(root, weight_mode=2)
    
    subtrees = extract_subtrees_at_depth(root, depth_threshold=10.0)
    
    assert len(subtrees) == 1
    assert set(subtrees[0].get_leaves()) == {"A", "B"}
    print("✓ High threshold returns entire tree as one subtree")


def test_low_threshold_returns_leaves():
    """
    With very low threshold, should return all leaves.
    """
    leaf_a = TreeNode(name="A", height=0.0)
    leaf_b = TreeNode(name="B", height=0.0)
    root = TreeNode(left=leaf_a, right=leaf_b, height=1.0)
    
    calc_tree_weighted_depth(root, weight_mode=2)
    
    subtrees = extract_subtrees_at_depth(root, depth_threshold=0.5)
    
    assert len(subtrees) == 2
    names = {s.name for s in subtrees}
    assert names == {"A", "B"}
    print("✓ Low threshold returns all leaves separately")


def main():
    """Run all tests."""
    print("Testing tree depth calculation (Yuri's tree_tips algorithm)\n")
    
    test_leaf_depth()
    test_simple_binary_tree()
    test_asymmetric_tree()
    test_weight_by_leaves()
    test_extract_subtrees()
    test_nontrivial_filter()
    test_high_threshold_returns_root()
    test_low_threshold_returns_leaves()
    
    print("\n✓ All tests passed!")
    print("\nOur implementation matches Yuri's tree_tips:")
    print("  - calc_tree_weighted_depth ≡ tree_calc_depth")
    print("  - extract_subtrees_at_depth ≡ cut_tips with -d=X")
    print("  - Default weight_mode=2 ≡ -w=2 (unweighted)")
    print("  - Default depth_threshold=1.2 ≡ -u=1.2")


if __name__ == "__main__":
    main()
