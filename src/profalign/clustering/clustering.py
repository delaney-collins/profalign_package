"""
clustering.py - Hierarchical clustering for profile alignment

Uses scipy.cluster.hierarchy to perform UPGMA (or other methods)
clustering on distance matrices and produces Newick trees.

This replaces the R-based run_R_hclust from the original pipeline.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree, ClusterNode
from scipy.spatial.distance import squareform
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass
from pathlib import Path
from enum import IntEnum


# Clustering method mapping (scipy names)
CLUSTERING_METHODS = {
    'upgma': 'average',      # UPGMA = Unweighted Pair Group Method with Arithmetic Mean
    'wpgma': 'weighted',     # WPGMA = Weighted Pair Group Method with Arithmetic Mean  
    'single': 'single',      # Single linkage (nearest neighbor)
    'complete': 'complete',  # Complete linkage (farthest neighbor)
    'ward': 'ward',          # Ward's method
}


class DepthWeightMode(IntEnum):
    """
    Weight modes for calculating tree depth (matches Yuri's -w parameter).
    
    The depth of a node is the weighted average distance to its leaves.
    """
    BY_LEAVES = 0    # Weight by number of leaves in each subtree
    BY_TBL = 1       # Weight by total branch length of each subtree
    UNWEIGHTED = 2   # Unweighted average (equal weight to each child)


def calc_weighted_depth(node: ClusterNode, 
                        names: List[str],
                        weight_mode: DepthWeightMode = DepthWeightMode.UNWEIGHTED,
                        _cache: Dict[int, Tuple[float, int, float]] = None) -> Tuple[float, int, float]:
    """
    Calculate weighted depth for a node (Yuri's tree_calc_depth).
    
    Depth is the weighted average distance from this node to its leaves.
    
    Args:
        node: ClusterNode from scipy
        names: List of leaf names
        weight_mode: How to weight children when averaging
        _cache: Internal cache for memoization
        
    Returns:
        Tuple of (depth, num_leaves, total_branch_length)
    """
    if _cache is None:
        _cache = {}
    
    # Check cache
    if node.id in _cache:
        return _cache[node.id]
    
    if node.is_leaf():
        # Leaves have depth 0, 1 leaf, 0 branch length below
        result = (0.0, 1, 0.0)
        _cache[node.id] = result
        return result
    
    # Get children info
    children_info = []
    for child in [node.left, node.right]:
        if child is not None:
            child_depth, child_leaves, child_tbl = calc_weighted_depth(
                child, names, weight_mode, _cache
            )
            # Branch length from this node to child
            branch_len = node.dist - (child.dist if not child.is_leaf() else 0)
            children_info.append((child_depth, child_leaves, child_tbl, branch_len))
    
    # Calculate weighted average depth
    weighted_sum = 0.0
    weight_sum = 0.0
    total_leaves = 0
    total_tbl = 0.0
    
    for child_depth, child_leaves, child_tbl, branch_len in children_info:
        # Determine weight based on mode
        if weight_mode == DepthWeightMode.BY_LEAVES:
            weight = child_leaves
        elif weight_mode == DepthWeightMode.BY_TBL:
            weight = child_tbl + branch_len if child_tbl > 0 else branch_len
        else:  # UNWEIGHTED
            weight = 1.0
        
        # Distance from this node through this child to leaves
        # = child's depth + branch length to child
        dist_to_leaves = child_depth + branch_len
        
        weighted_sum += dist_to_leaves * weight
        weight_sum += weight
        total_leaves += child_leaves
        total_tbl += child_tbl + branch_len
    
    depth = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    result = (depth, total_leaves, total_tbl)
    _cache[node.id] = result
    return result



def hierarchical_clustering(distance_matrix: np.ndarray,
                           method: str = 'upgma') -> np.ndarray:
    """
    Perform hierarchical clustering on a distance matrix.
    
    Args:
        distance_matrix: Square symmetric distance matrix (n x n)
        method: Clustering method - 'upgma', 'wpgma', 'single', 'complete', 'ward'
        
    Returns:
        Linkage matrix (scipy format)
    """
    # Convert method name
    scipy_method = CLUSTERING_METHODS.get(method.lower(), 'average')
    
    # scipy.linkage expects condensed distance matrix (upper triangle)
    # Ensure matrix is symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed form
    condensed = squareform(distance_matrix, checks=False)
    
    # Perform clustering
    linkage_matrix = linkage(condensed, method=scipy_method)
    
    return linkage_matrix


def linkage_to_newick(linkage_matrix: np.ndarray, 
                      names: List[str],
                      branch_lengths: bool = True) -> str:
    """
    Convert scipy linkage matrix to Newick tree format.
    
    Args:
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy.linkage
        names: List of leaf names (same order as distance matrix)
        branch_lengths: Include branch lengths in output (default: True)
        
    Returns:
        Newick format string
    """
    tree = to_tree(linkage_matrix)
    
    def node_to_newick(node: ClusterNode, parent_height: float = 0) -> str:
        """Recursively convert ClusterNode to Newick string."""
        if node.is_leaf():
            name = names[node.id]
            if branch_lengths:
                # Branch length = distance from this node to parent
                length = parent_height - 0  # Leaves are at height 0
                return f"{name}:{length:.6f}"
            else:
                return name
        else:
            # Internal node
            left_str = node_to_newick(node.left, node.dist)
            right_str = node_to_newick(node.right, node.dist)
            
            if branch_lengths:
                length = parent_height - node.dist if parent_height > 0 else 0
                return f"({left_str},{right_str}):{length:.6f}"
            else:
                return f"({left_str},{right_str})"
    
    # Get Newick string (root has no branch length)
    newick = node_to_newick(tree, tree.dist)
    
    # Remove trailing :0.000000 from root if present
    if branch_lengths and newick.endswith(':0.000000'):
        newick = newick[:-9]
    
    return newick + ';'


def build_tree_from_distance_matrix(distance_matrix: np.ndarray,
                                     names: List[str],
                                     method: str = 'upgma') -> str:
    """
    Build Newick tree from distance matrix.
    
    Convenience function combining clustering and tree conversion.
    
    Args:
        distance_matrix: Square symmetric distance matrix
        names: List of profile/sequence names
        method: Clustering method (default: 'upgma')
        
    Returns:
        Newick format tree string
    """
    linkage_matrix = hierarchical_clustering(distance_matrix, method)
    newick = linkage_to_newick(linkage_matrix, names)
    return newick


def get_tree_height(linkage_matrix: np.ndarray) -> float:
    """
    Get the total height (maximum distance) of the tree.
    
    Args:
        linkage_matrix: Linkage matrix from clustering
        
    Returns:
        Maximum height of the tree
    """
    if len(linkage_matrix) == 0:
        return 0.0
    return float(linkage_matrix[-1, 2])


def get_clusters_at_height(linkage_matrix: np.ndarray,
                           names: List[str],
                           height: float) -> List[List[str]]:
    """
    Get clusters by cutting the tree at a specific height.
    
    Args:
        linkage_matrix: Linkage matrix from clustering
        names: List of leaf names
        height: Height at which to cut the tree
        
    Returns:
        List of clusters, where each cluster is a list of names
    """
    from scipy.cluster.hierarchy import fcluster
    
    # fcluster returns cluster assignments (1-indexed)
    cluster_ids = fcluster(linkage_matrix, height, criterion='distance')
    
    # Group names by cluster
    clusters = {}
    for name, cluster_id in zip(names, cluster_ids):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(name)
    
    return list(clusters.values())


def get_subtrees_at_depth(linkage_matrix: np.ndarray,
                          names: List[str],
                          depth_threshold: float,
                          weight_mode: DepthWeightMode = DepthWeightMode.UNWEIGHTED) -> List[Tuple[List[str], float]]:
    """
    Get subtrees where root is within depth_threshold from leaves.
    
    This implements the tree_tips logic from the original pipeline:
    find subtrees suitable for profile merging based on their depth.
    
    Depth calculation uses weighted average (Yuri's algorithm):
    - BY_LEAVES (0): Weight by number of leaves in each subtree
    - BY_TBL (1): Weight by total branch length
    - UNWEIGHTED (2): Equal weight to each child (default)
    
    Args:
        linkage_matrix: Linkage matrix from clustering
        names: List of leaf names
        depth_threshold: Maximum depth (weighted avg distance to leaves) for subtree roots
        weight_mode: How to weight children when calculating depth
        
    Returns:
        List of (member_names, subtree_depth) tuples
    """
    tree = to_tree(linkage_matrix)
    subtrees = []
    
    # Pre-calculate weighted depths for all nodes
    depth_cache = {}
    calc_weighted_depth(tree, names, weight_mode, depth_cache)
    
    def get_node_depth(node: ClusterNode) -> float:
        """Get weighted depth for a node from cache."""
        if node.id in depth_cache:
            return depth_cache[node.id][0]
        return 0.0
    
    def collect_subtrees(node: ClusterNode):
        """
        Recursively collect subtrees at appropriate depth.
        
        A subtree is collected if:
        - It has more than 1 leaf (non-trivial)
        - Its weighted depth from leaves is <= depth_threshold
        """
        if node.is_leaf():
            return
        
        node_depth = get_node_depth(node)
        
        if node_depth <= depth_threshold:
            # This subtree is shallow enough - collect it
            members = get_leaves(node, names)
            if len(members) > 1:  # Non-trivial subtree
                subtrees.append((members, node_depth))
        else:
            # Too deep - recurse into children
            if node.left:
                collect_subtrees(node.left)
            if node.right:
                collect_subtrees(node.right)
    
    collect_subtrees(tree)
    
    # If no subtrees found (tree too shallow), return all leaves as singletons
    if not subtrees:
        subtrees = [([name], 0.0) for name in names]
    
    return subtrees


def get_leaves(node: ClusterNode, names: List[str]) -> List[str]:
    """
    Get all leaf names under a ClusterNode.
    
    Args:
        node: ClusterNode from scipy
        names: List of all leaf names
        
    Returns:
        List of leaf names in this subtree
    """
    if node.is_leaf():
        return [names[node.id]]
    
    leaves = []
    if node.left:
        leaves.extend(get_leaves(node.left, names))
    if node.right:
        leaves.extend(get_leaves(node.right, names))
    
    return leaves


