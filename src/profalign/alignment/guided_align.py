"""
guided_align.py - Tree-guided profile alignment (Yuri's algorithm)

This module implements the hh_guided_align logic:
1. Takes a guide tree (subtree from UPGMA clustering)
2. Recursively merges profiles following tree structure (post-order)
3. Returns single merged profile

Reference: Wolf Y.I. hh_guided_align script
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import shutil

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """
    Binary tree node for guide tree.
    
    Either a leaf (has name, no children) or internal node (has children, no name).
    """
    name: Optional[str] = None          # Leaf name (None for internal nodes)
    left: Optional['TreeNode'] = None   # Left child
    right: Optional['TreeNode'] = None  # Right child
    height: float = 0.0                 # Height from leaves (0 for leaves)
    weighted_depth: float = 0.0         # Weighted avg depth to leaves (Yuri's -w)
    
    def is_leaf(self) -> bool:
        return self.name is not None
    
    def get_leaves(self) -> List[str]:
        """Get all leaf names in this subtree."""
        if self.is_leaf():
            return [self.name]
        leaves = []
        if self.left:
            leaves.extend(self.left.get_leaves())
        if self.right:
            leaves.extend(self.right.get_leaves())
        return leaves
    
    def leaf_count(self) -> int:
        """Count leaves in this subtree."""
        return len(self.get_leaves())
    
    def total_branch_length(self) -> float:
        """Get total branch length below this node."""
        if self.is_leaf():
            return 0.0
        tbl = 0.0
        if self.left:
            branch_to_left = self.height - self.left.height
            tbl += branch_to_left + self.left.total_branch_length()
        if self.right:
            branch_to_right = self.height - self.right.height
            tbl += branch_to_right + self.right.total_branch_length()
        return tbl


def scipy_node_to_tree_node(node, names: List[str]) -> TreeNode:
    """
    Convert scipy ClusterNode to our TreeNode.
    
    Args:
        node: scipy.cluster.hierarchy ClusterNode
        names: List of leaf names
        
    Returns:
        TreeNode
    """
    if node.is_leaf():
        return TreeNode(name=names[node.id], height=0.0, weighted_depth=0.0)
    
    left_node = scipy_node_to_tree_node(node.left, names) if node.left else None
    right_node = scipy_node_to_tree_node(node.right, names) if node.right else None
    
    return TreeNode(
        left=left_node,
        right=right_node,
        height=node.dist,
        weighted_depth=0.0  # Will be calculated by calc_tree_weighted_depth
    )


def calc_tree_weighted_depth(node: TreeNode, weight_mode: int = 2) -> Tuple[float, int, float]:
    """
    Calculate weighted depth for a TreeNode.
    
    This is equivalent to Yuri's tree_calc_depth in tree_tips:
    - Depth = weighted average distance from this node DOWN to its leaves
    - NOT distance from root (that would be "height")
    - Computed bottom-up: leaves have depth=0, internal nodes average children
    
    The key insight: a node with LOW depth is close to its leaves,
    meaning it represents a "tight" cluster suitable for merging.
    
    Weight modes (Yuri's -w parameter in tree_tips):
        0: Weight by number of leaves in each subtree (BY_LEAVES)
        1: Weight by total branch length of each subtree (BY_TBL)
        2: Unweighted average (equal weight to each child) [default, matches -w=2]
    
    Note: Yuri's prof_align uses -w=2 (unweighted), which is our default.
    
    Yuri's Perl equivalent (from tree_tips):
        for each descendant:
            $lsum += ($$rden{"deep"} + $$rden{"l"}) * $dwei
            $wsum += $dwei
        $$rnod{"deep"} = $lsum / $wsum
    
    Args:
        node: TreeNode to calculate depth for
        weight_mode: How to weight children when averaging
        
    Returns:
        Tuple of (depth, num_leaves, total_branch_length)
    """
    if node.is_leaf():
        node.weighted_depth = 0.0
        return (0.0, 1, 0.0)
    
    # Get children info
    children_info = []
    for child in [node.left, node.right]:
        if child is not None:
            child_depth, child_leaves, child_tbl = calc_tree_weighted_depth(child, weight_mode)
            # Branch length from this node to child
            branch_len = node.height - child.height
            children_info.append((child_depth, child_leaves, child_tbl, branch_len))
    
    # Calculate weighted average depth
    weighted_sum = 0.0
    weight_sum = 0.0
    total_leaves = 0
    total_tbl = 0.0
    
    for child_depth, child_leaves, child_tbl, branch_len in children_info:
        # Determine weight based on mode
        if weight_mode == 0:  # BY_LEAVES
            weight = child_leaves
        elif weight_mode == 1:  # BY_TBL
            weight = child_tbl + branch_len if (child_tbl + branch_len) > 0 else 1.0
        else:  # UNWEIGHTED (mode 2)
            weight = 1.0
        
        # Distance from this node through this child to leaves
        # = child's depth + branch length to child
        dist_to_leaves = child_depth + branch_len
        
        weighted_sum += dist_to_leaves * weight
        weight_sum += weight
        total_leaves += child_leaves
        total_tbl += child_tbl + branch_len
    
    depth = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    node.weighted_depth = depth
    
    return (depth, total_leaves, total_tbl)


@dataclass
class GuidedMergeRecord:
    """Record of a single merge operation."""
    parent1: str          # Name of first parent (leaf or previous merge ID)
    parent2: str          # Name of second parent
    result_id: str        # Generated merge ID for result
    result_path: Path     # Path to result file


def run_guided_alignment(tree: TreeNode,
                         profile_paths: Dict[str, Path],
                         work_dir: Path,
                         merge_func,
                         id_generator,
                         gap_threshold: float = 0.667,
                         homogeneity_threshold: float = 0.0) -> Tuple[Path, str, List[GuidedMergeRecord]]:
    """
    Recursively merge profiles following guide tree structure.
    
    This implements hh_guided_align: post-order traversal where each
    internal node merges its children.
    
    Args:
        tree: Guide tree (TreeNode)
        profile_paths: Dict mapping profile name -> file path
        work_dir: Working directory for intermediate files
        merge_func: Function to merge two MSAs: merge_func(path1, path2, output_path) -> Alignment
        id_generator: Function to generate merge IDs: id_generator(name1, name2) -> str
        gap_threshold: Gap threshold for merge
        homogeneity_threshold: Homogeneity threshold for merge
        
    Returns:
        Tuple of:
        - Path to final merged alignment
        - Name/ID of final result (for tracking)
        - List of MergeRecord for each merge performed
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    merge_history: List[GuidedMergeRecord] = []
    # Track current name for each node (leaf name or merge ID)
    node_names: Dict[int, str] = {}
    node_paths: Dict[int, Path] = {}
    
    def merge_subtree(node: TreeNode) -> Tuple[Path, str]:
        """
        Recursively merge a subtree, returning (path, name) of result.
        """
        if node.is_leaf():
            # Leaf node - return existing profile
            if node.name not in profile_paths:
                raise ValueError(f"Profile not found: {node.name}")
            return profile_paths[node.name], node.name
        
        # Internal node - merge children
        # First, recursively get alignments for children
        left_path, left_name = merge_subtree(node.left) if node.left else (None, None)
        right_path, right_name = merge_subtree(node.right) if node.right else (None, None)
        
        if left_path is None or right_path is None:
            raise ValueError("Internal node must have both children")
        
        # Generate unique merge ID using the id_generator
        merge_id = id_generator(left_name, right_name)
        output_path = work_dir / f"{merge_id}.fasta"
        
        # Get names for logging
        left_display = left_name if node.left.is_leaf() else f"{left_name} [{node.left.leaf_count()} profiles]"
        right_display = right_name if node.right.is_leaf() else f"{right_name} [{node.right.leaf_count()} profiles]"
        
        logger.info(f"  Merging: {left_display} + {right_display} -> {merge_id}")
        
        # Perform merge
        try:
            merge_func(
                left_path,
                right_path,
                output_path,
                gap_threshold=gap_threshold,
                homogeneity_threshold=homogeneity_threshold
            )
        except Exception as e:
            logger.error(f"  Merge failed: {e}")
            raise
        
        # Record merge with proper names
        merge_history.append(GuidedMergeRecord(
            parent1=left_name,
            parent2=right_name,
            result_id=merge_id,
            result_path=output_path
        ))
        
        return output_path, merge_id
    
    # Run recursive merge
    final_path, final_name = merge_subtree(tree)
    
    return final_path, final_name, merge_history


def extract_subtrees_at_depth(tree: TreeNode, 
                               depth_threshold: float,
                               use_weighted_depth: bool = True) -> List[TreeNode]:
    """
    Extract subtrees where root depth <= depth_threshold.
    
    This implements Yuri's tree_tips -d=X logic (cut_tips function):
    - Walk the tree from root
    - When node.depth <= threshold, collect that subtree (it's "shallow" enough)
    - Otherwise, recurse into children
    
    The condition (depth <= threshold) matches Yuri's:
        if($dcut>0 and $$rnod{"deep"}<=$dcut) { ... output subtree ... }
    
    Intuition: 
    - Low depth = leaves are close = tight cluster = good to merge
    - High depth = leaves are far = loose cluster = need to split further
    
    Default threshold 1.2 matches Yuri's -u=1.2 in prof_align.
    
    Args:
        tree: Root of tree
        depth_threshold: Maximum depth for subtree roots (Yuri's -d= or -u=)
        use_weighted_depth: If True, use weighted_depth field (after calc_tree_weighted_depth).
                           If False, use height field (for -h= mode, rarely used).
        
    Returns:
        List of subtree roots (TreeNode objects)
    """
    subtrees = []
    
    def get_depth(node: TreeNode) -> float:
        """Get the appropriate depth value for a node."""
        if use_weighted_depth:
            return node.weighted_depth
        return node.height
    
    def collect(node: TreeNode):
        if node.is_leaf():
            # Single leaf - always return as trivial subtree
            subtrees.append(node)
            return
        
        node_depth = get_depth(node)
        
        if node_depth <= depth_threshold:
            # This subtree is shallow enough - collect it
            subtrees.append(node)
        else:
            # Too deep - recurse into children
            if node.left:
                collect(node.left)
            if node.right:
                collect(node.right)
    
    collect(tree)
    return subtrees


def get_nontrivial_subtrees(subtrees: List[TreeNode]) -> List[TreeNode]:
    """
    Filter to subtrees with >1 leaf.
    
    Args:
        subtrees: List of subtree roots
        
    Returns:
        List of non-trivial (>1 leaf) subtrees
    """
    return [s for s in subtrees if not s.is_leaf()]


def print_subtree(node: TreeNode, indent: int = 0):
    """Print tree structure for debugging."""
    prefix = "  " * indent
    if node.is_leaf():
        print(f"{prefix}└─ {node.name}")
    else:
        print(f"{prefix}├─ [h={node.height:.3f}]")
        if node.left:
            print_subtree(node.left, indent + 1)
        if node.right:
            print_subtree(node.right, indent + 1)
