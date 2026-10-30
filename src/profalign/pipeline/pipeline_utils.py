"""
pipeline_utils.py - Utilities for iterative profile alignment pipeline

This module provides:
1. Deterministic ID generation for merged profiles
2. Distance matrix storage (TSV format)
3. Merge history tracking

Design principles:
- All state stored in files (resumable after crashes)
- Short deterministic IDs (no collision risk in parallel execution)
- Human-readable formats for debugging
"""

import hashlib
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Merge ID Generation
# =============================================================================

def generate_merge_id(name1: str, name2: str) -> str:
    """
    Generate deterministic short ID for merged profile.
    
    The ID is a hash of the sorted pair of names, ensuring:
    - merge(A, B) == merge(B, A) (same ID)
    - Unique for each pair (12 hex chars = 48 bits, collision at ~16M merges)
    - No collision risk in parallel execution
    
    Args:
        name1: First profile name
        name2: Second profile name
        
    Returns:
        Merge ID in format "M_xxxxxxxxxxxx" (12 hex chars)
    """
    # Sort for consistency
    pair = "_".join(sorted([name1, name2]))
    hash_hex = hashlib.md5(pair.encode()).hexdigest()[:12]
    return f"M_{hash_hex}"


@dataclass
class MergeRecord:
    """Record of a single merge operation."""
    merge_id: str
    parent1: str
    parent2: str
    iteration: int
    
    def to_tsv_row(self) -> str:
        return f"{self.merge_id}\t{self.parent1}\t{self.parent2}\t{self.iteration}"
    
    @classmethod
    def from_tsv_row(cls, row: List[str]) -> 'MergeRecord':
        return cls(
            merge_id=row[0],
            parent1=row[1],
            parent2=row[2],
            iteration=int(row[3])
        )


class MergeHistory:
    """
    Tracks merge operations across iterations.
    
    Stores mapping: merge_id -> (parent1, parent2, iteration)
    Allows tracing any merged profile back to original inputs.
    """
    
    HEADER = "# merge_id\tparent1\tparent2\titeration\n"
    
    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = Path(filepath) if filepath else None
        self.records: Dict[str, MergeRecord] = {}
        
        if self.filepath and self.filepath.exists():
            self.load()
    
    def add(self, merge_id: str, parent1: str, parent2: str, iteration: int):
        """Add a merge record."""
        record = MergeRecord(merge_id, parent1, parent2, iteration)
        self.records[merge_id] = record
        
        # Append to file immediately (crash-safe)
        if self.filepath:
            self._append_record(record)
    
    def get(self, merge_id: str) -> Optional[MergeRecord]:
        """Get merge record by ID."""
        return self.records.get(merge_id)
    
    def get_parents(self, merge_id: str) -> Optional[Tuple[str, str]]:
        """Get parent names for a merge ID."""
        record = self.records.get(merge_id)
        if record:
            return (record.parent1, record.parent2)
        return None
    
    def get_original_profiles(self, profile_id: str) -> List[str]:
        """
        Recursively trace back to original (non-merged) profiles.
        
        Args:
            profile_id: Any profile ID (original or merged)
            
        Returns:
            List of original profile names that were merged to create this
        """
        record = self.records.get(profile_id)
        if record is None:
            # Not a merge - this is an original profile
            return [profile_id]
        
        # Recursively get originals from both parents
        originals = []
        originals.extend(self.get_original_profiles(record.parent1))
        originals.extend(self.get_original_profiles(record.parent2))
        return originals
    
    def load(self):
        """Load history from file."""
        if not self.filepath or not self.filepath.exists():
            return
        
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    record = MergeRecord.from_tsv_row(parts)
                    self.records[record.merge_id] = record
        
        logger.info(f"Loaded {len(self.records)} merge records from {self.filepath}")
    
    def save(self):
        """Save entire history to file (overwrites)."""
        if not self.filepath:
            return
        
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.filepath, 'w') as f:
            f.write(self.HEADER)
            for record in self.records.values():
                f.write(record.to_tsv_row() + '\n')
    
    def _append_record(self, record: MergeRecord):
        """Append single record to file."""
        if not self.filepath:
            return
        
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with header if doesn't exist
        if not self.filepath.exists():
            with open(self.filepath, 'w') as f:
                f.write(self.HEADER)
        
        with open(self.filepath, 'a') as f:
            f.write(record.to_tsv_row() + '\n')


# =============================================================================
# 2. Distance Matrix Storage
# =============================================================================

@dataclass
class ScorePair:
    """
    Single pairwise score measurement.
    
    Stores raw HHsearch score for later distance calculation.
    Distance is computed from scores using normalization and symmetrization.
    """
    profile_a: str
    profile_b: str
    score: float          # Raw HHsearch score (NOT normalized distance!)
    probability: float    # HHsearch probability (0-100)
    aligned_cols: int     # Number of aligned columns
    coverage: float = 0.0 # Min coverage of query/template (0-1)
    
    def to_tsv_row(self) -> str:
        return f"{self.profile_a}\t{self.profile_b}\t{self.score:.6f}\t{self.probability:.2f}\t{self.aligned_cols}\t{self.coverage:.4f}"
    
    @classmethod
    def from_tsv_row(cls, row: List[str]) -> 'ScorePair':
        # Backward compatible: coverage may be missing in old files
        coverage = float(row[5]) if len(row) > 5 else 0.0
        return cls(
            profile_a=row[0],
            profile_b=row[1],
            score=float(row[2]),
            probability=float(row[3]),
            aligned_cols=int(row[4]),
            coverage=coverage
        )
    
    def key(self) -> Tuple[str, str]:
        """Return key for lookup (NOT sorted - direction matters for scores!)."""
        return (self.profile_a, self.profile_b)
    
    def symmetric_key(self) -> Tuple[str, str]:
        """Return sorted key for symmetric lookup."""
        return tuple(sorted([self.profile_a, self.profile_b]))


class ScoreStore:
    """
    Storage for pairwise scores.
    
    Stores RAW scores from HHsearch. Distance matrix is computed later
    using normalization and symmetrization (Yuri's algorithm).
    
    Features:
    - TSV file format (human-readable, appendable)
    - Directional storage: score(A,B) may differ from score(B,A)
    - Includes self-scores: score(A,A) for normalization
    - Incremental updates (append new pairs)
    """
    
    HEADER = "# profile_a\tprofile_b\tscore\tprobability\taligned_cols\tcoverage\n"
    
    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = Path(filepath) if filepath else None
        # Store by directional key (a, b) - NOT symmetric!
        self.pairs: Dict[Tuple[str, str], ScorePair] = {}
        
        if self.filepath and self.filepath.exists():
            self.load()
    
    def add(self, profile_a: str, profile_b: str, score: float,
            probability: float = 0.0, aligned_cols: int = 0,
            coverage: float = 0.0,
            append_to_file: bool = True):
        """
        Add a score pair.
        
        Note: Stores directionally! score(A,B) and score(B,A) are separate.
        
        Args:
            profile_a, profile_b: Profile names (a=query, b=template)
            score: Raw HHsearch score
            probability: HHsearch probability
            aligned_cols: Number of aligned columns
            coverage: Min coverage of query/template (0-1)
            append_to_file: If True, immediately append to file
        """
        pair = ScorePair(profile_a, profile_b, score, probability, aligned_cols, coverage)
        key = pair.key()  # Directional: (a, b)
        self.pairs[key] = pair
        
        if append_to_file and self.filepath:
            self._append_pair(pair)
    
    def get(self, profile_a: str, profile_b: str) -> Optional[ScorePair]:
        """Get score pair (directional - order matters!)."""
        key = (profile_a, profile_b)
        return self.pairs.get(key)
    
    def get_score(self, profile_a: str, profile_b: str, 
                  default: float = 0.0) -> float:
        """Get just the score value (directional)."""
        pair = self.get(profile_a, profile_b)
        return pair.score if pair else default
    
    def get_self_score(self, profile: str, default: float = 0.0) -> float:
        """Get self-score for a profile (A vs A)."""
        return self.get_score(profile, profile, default)
    
    def has_pair(self, profile_a: str, profile_b: str) -> bool:
        """Check if score exists for this pair (directional)."""
        key = (profile_a, profile_b)
        return key in self.pairs
    
    def has_symmetric_pair(self, profile_a: str, profile_b: str) -> bool:
        """Check if score exists in either direction."""
        return self.has_pair(profile_a, profile_b) or self.has_pair(profile_b, profile_a)
    
    def get_all_profiles(self) -> Set[str]:
        """Get set of all profile names in the store."""
        profiles = set()
        for pair in self.pairs.values():
            profiles.add(pair.profile_a)
            profiles.add(pair.profile_b)
        return profiles
    
    def get_pairs_for_profile(self, profile: str) -> List[ScorePair]:
        """Get all pairs where profile is query."""
        result = []
        for pair in self.pairs.values():
            if pair.profile_a == profile:
                result.append(pair)
        return result
    
    def remove_profile(self, profile: str):
        """Remove all pairs involving a profile (after merge)."""
        keys_to_remove = []
        for key, pair in self.pairs.items():
            if pair.profile_a == profile or pair.profile_b == profile:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.pairs[key]
    
    def load(self):
        """Load scores from file."""
        if not self.filepath or not self.filepath.exists():
            return
        
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 5:
                    pair = ScorePair.from_tsv_row(parts)
                    self.pairs[pair.key()] = pair
        
        logger.info(f"Loaded {len(self.pairs)} score pairs from {self.filepath}")
    
    def save(self):
        """Save all scores to file (overwrites)."""
        if not self.filepath:
            return
        
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.filepath, 'w') as f:
            f.write(self.HEADER)
            for pair in self.pairs.values():
                f.write(pair.to_tsv_row() + '\n')
    
    def _append_pair(self, pair: ScorePair):
        """Append single pair to file."""
        if not self.filepath:
            return
        
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.filepath.exists():
            with open(self.filepath, 'w') as f:
                f.write(self.HEADER)
        
        with open(self.filepath, 'a') as f:
            f.write(pair.to_tsv_row() + '\n')
    
    def to_task_results(self) -> List:
        """
        Convert stored scores to TaskResult-like objects for distance_matrix.
        
        Returns list of objects with .query_name, .target_name, .score, .aligned_cols, .success
        """
        from dataclasses import dataclass
        
        @dataclass
        class _Result:
            query_name: str
            target_name: str
            score: float
            aligned_cols: int = 0
            success: bool = True
        
        return [_Result(p.profile_a, p.profile_b, p.score, p.aligned_cols) for p in self.pairs.values()]


# Backward compatibility aliases
DistanceStore = ScoreStore
DistancePair = ScorePair


# =============================================================================
# 3. Helper Functions
# =============================================================================

def find_missing_pairs(profiles: List[str], 
                       existing_store: ScoreStore,
                       include_self: bool = True) -> List[Tuple[str, str]]:
    """
    Find all profile pairs that don't have scores calculated yet.
    
    IMPORTANT: Includes self-comparisons (A vs A) for normalization!
    Also includes both directions (A vs B) and (B vs A) for asymmetric scores.
    
    Args:
        profiles: List of all current profile names
        existing_store: Store with already calculated scores
        include_self: If True, include self-comparisons (A vs A)
        
    Returns:
        List of (query, template) pairs that need calculation
    """
    missing = []
    n = len(profiles)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # Self-comparison (A vs A)
                if include_self and not existing_store.has_pair(profiles[i], profiles[i]):
                    missing.append((profiles[i], profiles[i]))
            else:
                # Cross-comparison (A vs B) - check directionally
                if not existing_store.has_pair(profiles[i], profiles[j]):
                    missing.append((profiles[i], profiles[j]))
    
    return missing


def find_pairs_for_new_profiles(new_profiles: List[str],
                                all_profiles: List[str],
                                existing_store: DistanceStore) -> List[Tuple[str, str]]:
    """
    Find pairs to calculate for newly created profiles.
    
    New profiles need distances to:
    - All other profiles (including other new ones)
    - Excluding pairs that somehow already exist
    
    Args:
        new_profiles: List of newly created profile names
        all_profiles: List of all profile names (including new)
        existing_store: Store with already calculated distances
        
    Returns:
        List of (profile_a, profile_b) pairs to calculate
    """
    pairs = []
    new_set = set(new_profiles)
    
    for new_prof in new_profiles:
        for other_prof in all_profiles:
            if new_prof == other_prof:
                continue
            
            # Avoid duplicates: only add if new_prof < other_prof or other is not new
            if other_prof in new_set and other_prof < new_prof:
                continue
            
            if not existing_store.has_pair(new_prof, other_prof):
                pairs.append((new_prof, other_prof))
    
    return pairs
