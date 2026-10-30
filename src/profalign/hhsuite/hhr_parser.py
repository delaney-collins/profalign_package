"""
hhr_parser.py - Parser for HHsearch result files (.hhr)

Parses .hhr files produced by hhsearch and extracts hit information
to build score/distance matrices for clustering.

Based on the hit table format in HHsearch 3.x output:
    No Hit                             Prob E-value P-value  Score    SS Cols Query HMM  Template HMM
     1 profile_name                   100.0  1E-100  1E-105  500.0   0.0  200   1-200     1-200 (200)
"""

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class HHRHit:
    """Single hit from HHsearch results."""
    rank: int
    target_id: str
    probability: float      # Probability (0-100)
    evalue: float
    pvalue: float
    score: float            # Raw score
    ss_score: float         # Secondary structure score
    aligned_cols: int       # Number of aligned columns
    query_start: int
    query_end: int
    template_start: int
    template_end: int
    template_length: int


@dataclass  
class HHRResult:
    """Parsed HHsearch result file."""
    query_id: str
    query_length: int
    hits: List[HHRHit]


def parse_hhr_file(filepath: Union[str, Path]) -> HHRResult:
    """
    Parse a single .hhr file.
    
    Args:
        filepath: Path to .hhr file
        
    Returns:
        HHRResult object with query info and list of hits
    """
    filepath = Path(filepath)
    
    query_id = None
    query_length = None
    hits = []
    
    # Regex for parsing hit lines
    # Example: "  1 profile_name   100.0  1E-100  1E-105  500.0   0.0  200   1-200     1-200 (200)"
    hit_pattern = re.compile(
        r'^\s*(\d+)\s+'           # Rank
        r'(\S+)\s+'               # Target ID
        r'(\d+\.?\d*)\s+'         # Probability
        r'(\S+)\s+'               # E-value
        r'(\S+)\s+'               # P-value
        r'(\S+)\s+'               # Score
        r'(\S+)\s+'               # SS score
        r'(\d+)\s+'               # Aligned columns
        r'(\d+)-(\d+)\s+'         # Query range
        r'(\d+)-(\d+)\s*'         # Template range (space optional before length)
        r'\((\d+)\)'              # Template length
    )
    
    with open(filepath, 'r') as f:
        in_hit_table = False
        
        for line in f:
            line = line.rstrip()
            
            # Parse query info
            if line.startswith('Query'):
                parts = line.split(None, 1)
                if len(parts) > 1:
                    query_id = parts[1].split()[0]
                    
            elif line.startswith('Match_columns'):
                parts = line.split()
                if len(parts) > 1:
                    query_length = int(parts[1])
                    
            # Detect start of hit table
            elif line.strip().startswith('No Hit'):
                in_hit_table = True
                continue
                
            # Parse hit lines
            elif in_hit_table:
                # Empty line or new section ends hit table
                if not line.strip() or line.startswith('>'):
                    in_hit_table = False
                    continue
                
                match = hit_pattern.match(line)
                if match:
                    hit = HHRHit(
                        rank=int(match.group(1)),
                        target_id=match.group(2),
                        probability=float(match.group(3)),
                        evalue=float(match.group(4)),
                        pvalue=float(match.group(5)),
                        score=float(match.group(6)),
                        ss_score=float(match.group(7)),
                        aligned_cols=int(match.group(8)),
                        query_start=int(match.group(9)),
                        query_end=int(match.group(10)),
                        template_start=int(match.group(11)),
                        template_end=int(match.group(12)),
                        template_length=int(match.group(13))
                    )
                    hits.append(hit)
    
    return HHRResult(
        query_id=query_id or filepath.stem,
        query_length=query_length or 0,
        hits=hits
    )


def parse_hhr_string(content: str, default_query: str = "unknown") -> HHRResult:
    """
    Parse HHR content from a string (e.g., downloaded from cloud storage).

    Args:
        content: HHR file content as string
        default_query: Fallback query name if not found in content

    Returns:
        HHRResult object with query info and list of hits
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile('w', suffix='.hhr', delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = parse_hhr_file(tmp_path)
        if result.query_id == Path(tmp_path).stem:
            result = HHRResult(
                query_id=default_query,
                query_length=result.query_length,
                hits=result.hits
            )
        return result
    finally:
        os.unlink(tmp_path)


def parse_multiple_hhr(hhr_files: List[Union[str, Path]]) -> Dict[str, HHRResult]:
    """
    Parse multiple .hhr files.
    
    Args:
        hhr_files: List of paths to .hhr files
        
    Returns:
        Dictionary mapping query_id to HHRResult
    """
    results = {}
    for hhr_file in hhr_files:
        result = parse_hhr_file(hhr_file)
        results[result.query_id] = result
    return results


def build_score_matrix(hhr_results: Dict[str, HHRResult],
                       score_type: str = 'score',
                       min_coverage: float = 0.0) -> Tuple[np.ndarray, List[str]]:
    """
    Build pairwise score matrix from HHsearch results.
    
    Args:
        hhr_results: Dictionary of parsed HHR results
        score_type: Which score to use - 'score', 'probability', 'evalue'
        min_coverage: Minimum alignment coverage (0-1) to consider a hit valid
        
    Returns:
        Tuple of (score_matrix, profile_names)
    """
    profile_names = sorted(hhr_results.keys())
    n = len(profile_names)
    name_to_idx = {name: i for i, name in enumerate(profile_names)}
    
    matrix = np.zeros((n, n))
    
    for query_name, result in hhr_results.items():
        query_idx = name_to_idx[query_name]
        query_len = result.query_length if result.query_length > 0 else 1
        
        for hit in result.hits:
            if hit.target_id not in name_to_idx:
                continue
            
            target_idx = name_to_idx[hit.target_id]
            
            coverage = hit.aligned_cols / query_len
            if coverage < min_coverage:
                continue
            
            if score_type == 'score':
                value = hit.score
            elif score_type == 'probability':
                value = hit.probability / 100.0
            elif score_type == 'evalue':
                value = hit.evalue
            else:
                value = hit.score
            
            if score_type == 'evalue':
                if matrix[query_idx, target_idx] == 0 or value < matrix[query_idx, target_idx]:
                    matrix[query_idx, target_idx] = value
            else:
                if value > matrix[query_idx, target_idx]:
                    matrix[query_idx, target_idx] = value
    
    return matrix, profile_names


def build_distance_matrix(hhr_results: Dict[str, HHRResult],
                          min_coverage: float = 0.0,
                          normalize: bool = True,
                          symmetrize: str = 'max',
                          distance_type: str = 'log',
                          max_distance: float = 12.0) -> Tuple[np.ndarray, List[str]]:
    """
    Build pairwise distance matrix from HHsearch results.
    
    This replicates the logic from Yuri's hhres2dis_mat script:
    1. Extract scores from all-vs-all HHsearch results
    2. Optionally normalize by self-scores
    3. Symmetrize the matrix
    4. Convert scores to distances
    
    Args:
        hhr_results: Dictionary of parsed HHR results
        min_coverage: Minimum alignment coverage (0-1) to consider hit valid
        normalize: Normalize scores by self-scores (default: True)
        symmetrize: How to symmetrize - 'max', 'min', or 'none' (default: 'max')
        distance_type: Distance transformation - 'log' (-ln(s)) or 'linear' (1-s)
        max_distance: Maximum distance for log transform (default: 12.0)
        
    Returns:
        Tuple of (distance_matrix, profile_names)
    """
    # First build raw score matrix
    score_matrix, profile_names = build_score_matrix(
        hhr_results, score_type='score', min_coverage=min_coverage
    )
    n = len(profile_names)
    
    # Normalize by self-scores
    if normalize:
        self_scores = np.diag(score_matrix).copy()
        self_scores[self_scores <= 0] = 1  # Avoid division by zero
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Normalize by query self-score (option 3 in original)
                    norm_factor = self_scores[i]
                    if norm_factor > 0:
                        score_matrix[i, j] /= norm_factor
        
        # Set diagonal to 1.0
        np.fill_diagonal(score_matrix, 1.0)
    
    # Symmetrize
    if symmetrize == 'max':
        score_matrix = np.maximum(score_matrix, score_matrix.T)
    elif symmetrize == 'min':
        score_matrix = np.minimum(score_matrix, score_matrix.T)
    # else 'none' - keep asymmetric
    
    # Convert to distance
    epsilon = math.exp(-max_distance)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                s = score_matrix[i, j]
                if distance_type == 'log':
                    if s > epsilon:
                        distance_matrix[i, j] = -math.log(s)
                    else:
                        distance_matrix[i, j] = max_distance
                else:  # linear
                    distance_matrix[i, j] = 1.0 - s
                    
                distance_matrix[i, j] = abs(distance_matrix[i, j])
    
    return distance_matrix, profile_names


def apply_name_mapping(hhr_results: Dict[str, HHRResult],
                       name_mapping: Dict[str, str]) -> Dict[str, HHRResult]:
    """
    Apply name mapping to HHR results.
    
    Converts target_ids from first sequence names (e.g., S00035_Calpl_1)
    to profile names (e.g., C25_Calpl_1) using the provided mapping.
    
    Args:
        hhr_results: Dictionary of query_id -> HHRResult
        name_mapping: Dictionary of first_seq_id -> profile_name
        
    Returns:
        New dictionary with mapped names
    """
    mapped_results = {}
    
    for query_id, result in hhr_results.items():
        # Map hits target_ids
        mapped_hits = []
        for hit in result.hits:
            mapped_target = name_mapping.get(hit.target_id, hit.target_id)
            mapped_hit = HHRHit(
                rank=hit.rank,
                target_id=mapped_target,
                probability=hit.probability,
                evalue=hit.evalue,
                pvalue=hit.pvalue,
                score=hit.score,
                ss_score=hit.ss_score,
                aligned_cols=hit.aligned_cols,
                query_start=hit.query_start,
                query_end=hit.query_end,
                template_start=hit.template_start,
                template_end=hit.template_end,
                template_length=hit.template_length
            )
            mapped_hits.append(mapped_hit)
        
        mapped_results[query_id] = HHRResult(
            query_id=result.query_id,
            query_length=result.query_length,
            hits=mapped_hits
        )
    
    return mapped_results
