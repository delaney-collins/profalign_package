"""
merge.py - Profile-guided MSA merging (Yuri's algorithm)

This module implements a careful MSA merging strategy that:
1. Filters columns by quality (gaps, homogeneity)
2. Uses hhalign only on "skeleton" alignments (good columns)
3. Reconstructs full alignment by re-inserting filtered columns

This approach preserves alignment quality better than naive hhalign
on full MSAs, especially for diverse sequences.

Reference: Wolf Y.I. (unpublished), as described in prof_align pipeline
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Handle imports for both package and direct script usage
from profalign.io.formats import Alignment, AlignedSequence, read_fasta, write_fasta

logger = logging.getLogger(__name__)


# BLOSUM62 similarity matrix for homogeneity calculation
BLOSUM62 = {
    'AA': 4, 'AR': -1, 'AN': -1, 'AD': -2, 'AC': 0, 'AQ': -1, 'AE': -1, 'AG': 0, 'AH': -2, 'AI': -1,
    'AL': -1, 'AK': -1, 'AM': -1, 'AF': -2, 'AP': -1, 'AS': 1, 'AT': 0, 'AW': -3, 'AY': -2, 'AV': 0,
    'RA': -1, 'RR': 5, 'RN': 0, 'RD': -1, 'RC': -3, 'RQ': 1, 'RE': 0, 'RG': -2, 'RH': 0, 'RI': -3,
    'RL': -2, 'RK': 2, 'RM': -1, 'RF': -3, 'RP': -2, 'RS': -1, 'RT': -1, 'RW': -3, 'RY': -2, 'RV': -2,
    'NA': -1, 'NR': 0, 'NN': 6, 'ND': 1, 'NC': -2, 'NQ': 0, 'NE': 0, 'NG': 0, 'NH': 1, 'NI': -3,
    'NL': -3, 'NK': 0, 'NM': -2, 'NF': -3, 'NP': -2, 'NS': 1, 'NT': 0, 'NW': -4, 'NY': -2, 'NV': -3,
    'DA': -2, 'DR': -1, 'DN': 1, 'DD': 6, 'DC': -3, 'DQ': 0, 'DE': 2, 'DG': -1, 'DH': -1, 'DI': -3,
    'DL': -3, 'DK': -1, 'DM': -3, 'DF': -3, 'DP': -1, 'DS': 0, 'DT': -1, 'DW': -4, 'DY': -3, 'DV': -3,
    'CA': 0, 'CR': -3, 'CN': -2, 'CD': -3, 'CC': 9, 'CQ': -3, 'CE': -3, 'CG': -2, 'CH': -3, 'CI': -1,
    'CL': -1, 'CK': -3, 'CM': -1, 'CF': -2, 'CP': -3, 'CS': -1, 'CT': -1, 'CW': -2, 'CY': -2, 'CV': -1,
    'QA': -1, 'QR': 1, 'QN': 0, 'QD': 0, 'QC': -3, 'QQ': 5, 'QE': 2, 'QG': -2, 'QH': 1, 'QI': -3,
    'QL': -2, 'QK': 1, 'QM': 0, 'QF': -3, 'QP': -1, 'QS': 0, 'QT': -1, 'QW': -2, 'QY': -1, 'QV': -2,
    'EA': -1, 'ER': 0, 'EN': 0, 'ED': 2, 'EC': -3, 'EQ': 2, 'EE': 5, 'EG': -2, 'EH': 0, 'EI': -3,
    'EL': -3, 'EK': 1, 'EM': -2, 'EF': -3, 'EP': -1, 'ES': 0, 'ET': -1, 'EW': -3, 'EY': -2, 'EV': -2,
    'GA': 0, 'GR': -2, 'GN': 0, 'GD': -1, 'GC': -2, 'GQ': -2, 'GE': -2, 'GG': 6, 'GH': -2, 'GI': -3,
    'GL': -4, 'GK': -1, 'GM': -2, 'GF': -3, 'GP': -2, 'GS': 0, 'GT': -2, 'GW': -2, 'GY': -3, 'GV': -3,
    'HA': -2, 'HR': 0, 'HN': 1, 'HD': -1, 'HC': -3, 'HQ': 1, 'HE': 0, 'HG': -2, 'HH': 7, 'HI': -3,
    'HL': -3, 'HK': -1, 'HM': -1, 'HF': -1, 'HP': -2, 'HS': -1, 'HT': -2, 'HW': -2, 'HY': 2, 'HV': -3,
    'IA': -1, 'IR': -3, 'IN': -3, 'ID': -3, 'IC': -1, 'IQ': -3, 'IE': -3, 'IG': -3, 'IH': -3, 'II': 4,
    'IL': 2, 'IK': -3, 'IM': 1, 'IF': 0, 'IP': -3, 'IS': -2, 'IT': -1, 'IW': -2, 'IY': -1, 'IV': 3,
    'LA': -1, 'LR': -2, 'LN': -3, 'LD': -3, 'LC': -1, 'LQ': -2, 'LE': -3, 'LG': -4, 'LH': -3, 'LI': 2,
    'LL': 4, 'LK': -2, 'LM': 2, 'LF': 0, 'LP': -3, 'LS': -2, 'LT': -1, 'LW': -2, 'LY': -1, 'LV': 1,
    'KA': -1, 'KR': 2, 'KN': 0, 'KD': -1, 'KC': -3, 'KQ': 1, 'KE': 1, 'KG': -1, 'KH': -1, 'KI': -3,
    'KL': -2, 'KK': 4, 'KM': -1, 'KF': -3, 'KP': -1, 'KS': 0, 'KT': -1, 'KW': -3, 'KY': -2, 'KV': -2,
    'MA': -1, 'MR': -1, 'MN': -2, 'MD': -3, 'MC': -1, 'MQ': 0, 'ME': -2, 'MG': -2, 'MH': -1, 'MI': 1,
    'ML': 2, 'MK': -1, 'MM': 5, 'MF': 0, 'MP': -2, 'MS': -1, 'MT': -1, 'MW': -1, 'MY': -1, 'MV': 1,
    'FA': -2, 'FR': -3, 'FN': -3, 'FD': -3, 'FC': -2, 'FQ': -3, 'FE': -3, 'FG': -3, 'FH': -1, 'FI': 0,
    'FL': 0, 'FK': -3, 'FM': 0, 'FF': 6, 'FP': -4, 'FS': -2, 'FT': -2, 'FW': 1, 'FY': 3, 'FV': -1,
    'PA': -1, 'PR': -2, 'PN': -2, 'PD': -1, 'PC': -3, 'PQ': -1, 'PE': -1, 'PG': -2, 'PH': -2, 'PI': -3,
    'PL': -3, 'PK': -1, 'PM': -2, 'PF': -4, 'PP': 7, 'PS': -1, 'PT': -1, 'PW': -4, 'PY': -3, 'PV': -2,
    'SA': 1, 'SR': -1, 'SN': 1, 'SD': 0, 'SC': -1, 'SQ': 0, 'SE': 0, 'SG': 0, 'SH': -1, 'SI': -2,
    'SL': -2, 'SK': 0, 'SM': -1, 'SF': -2, 'SP': -1, 'SS': 4, 'ST': 1, 'SW': -3, 'SY': -2, 'SV': -2,
    'TA': 0, 'TR': -1, 'TN': 0, 'TD': -1, 'TC': -1, 'TQ': -1, 'TE': -1, 'TG': -2, 'TH': -2, 'TI': -1,
    'TL': -1, 'TK': -1, 'TM': -1, 'TF': -2, 'TP': -1, 'TS': 1, 'TT': 4, 'TW': -2, 'TY': -2, 'TV': 0,
    'WA': -3, 'WR': -3, 'WN': -4, 'WD': -4, 'WC': -2, 'WQ': -2, 'WE': -3, 'WG': -2, 'WH': -2, 'WI': -2,
    'WL': -2, 'WK': -3, 'WM': -1, 'WF': 1, 'WP': -4, 'WS': -3, 'WT': -2, 'WW': 10, 'WY': 2, 'WV': -3,
    'YA': -2, 'YR': -2, 'YN': -2, 'YD': -3, 'YC': -2, 'YQ': -1, 'YE': -2, 'YG': -3, 'YH': 2, 'YI': -1,
    'YL': -1, 'YK': -2, 'YM': -1, 'YF': 3, 'YP': -3, 'YS': -2, 'YT': -2, 'YW': 2, 'YY': 6, 'YV': -1,
    'VA': 0, 'VR': -2, 'VN': -3, 'VD': -3, 'VC': -1, 'VQ': -2, 'VE': -2, 'VG': -3, 'VH': -3, 'VI': 3,
    'VL': 1, 'VK': -2, 'VM': 1, 'VF': -1, 'VP': -2, 'VS': -2, 'VT': 0, 'VW': -3, 'VY': -1, 'VV': 4,
}

# Background amino acid frequencies (from sr_filter)
AA_FREQ = {
    'A': 0.07422, 'C': 0.02469, 'D': 0.05363, 'E': 0.05431, 'F': 0.04742,
    'G': 0.07415, 'H': 0.02621, 'I': 0.06792, 'K': 0.05816, 'L': 0.09891,
    'M': 0.02499, 'N': 0.04465, 'P': 0.03854, 'Q': 0.03426, 'R': 0.05161,
    'S': 0.05723, 'T': 0.05089, 'V': 0.07292, 'W': 0.01303, 'Y': 0.03228,
}

# Precompute expected scores for each amino acid
AA_EXPECTED = {}
for aa in AA_FREQ:
    expected = sum(BLOSUM62.get(aa + bb, 0) * freq for bb, freq in AA_FREQ.items())
    AA_EXPECTED[aa] = expected


@dataclass
class ColumnStats:
    """Statistics for a single alignment column."""
    position: int           # 0-based position in alignment
    num_seqs: int           # Total number of sequences
    num_gaps: int           # Number of gaps in column
    gap_fraction: float     # Fraction of gaps
    homogeneity: float      # Column homogeneity score (0-1)
    consensus: str          # Consensus amino acid


@dataclass
class FilteredAlignment:
    """Alignment with column filtering information."""
    alignment: Alignment            # Original full alignment
    column_stats: List[ColumnStats] # Stats for each column
    filtered_positions: List[int]   # Positions that passed filter (0-based)
    full_to_filtered: Dict[int, int]  # Map full position -> filtered position (-1 if filtered out)
    filtered_to_full: Dict[int, int]  # Map filtered position -> full position


def compute_column_stats(alignment: Alignment) -> List[ColumnStats]:
    """
    Compute statistics for each column of an alignment.
    
    Args:
        alignment: Input alignment
        
    Returns:
        List of ColumnStats for each column
    """
    if not alignment.sequences:
        return []
    
    num_seqs = len(alignment.sequences)
    ali_length = len(alignment.sequences[0].sequence)
    stats = []
    
    for pos in range(ali_length):
        # Extract column
        column = [seq.sequence[pos].upper() for seq in alignment.sequences]
        
        # Count gaps
        num_gaps = sum(1 for c in column if c in '-.*')
        gap_fraction = num_gaps / num_seqs
        
        # Count amino acid frequencies
        aa_counts = {}
        non_gap_count = 0
        for c in column:
            if c in AA_FREQ:
                aa_counts[c] = aa_counts.get(c, 0) + 1
                non_gap_count += 1
        
        # Compute homogeneity using BLOSUM62
        homogeneity = 0.0
        consensus = '-'
        
        if non_gap_count > 0:
            # Find best consensus amino acid
            best_aa = None
            best_score = float('-inf')
            
            for aa in AA_FREQ:
                # Compute weighted score for this AA being consensus
                score = 0.0
                for obs_aa, count in aa_counts.items():
                    key = aa + obs_aa
                    if key in BLOSUM62:
                        score += BLOSUM62[key] * count
                
                if score > best_score:
                    best_score = score
                    best_aa = aa
            
            if best_aa:
                consensus = best_aa
                # Normalize homogeneity to 0-1 range
                # homogeneity = (observed - expected) / (max - expected)
                expected = AA_EXPECTED.get(best_aa, 0) * non_gap_count
                max_score = BLOSUM62.get(best_aa + best_aa, 4) * non_gap_count
                
                if max_score > expected:
                    homogeneity = (best_score - expected) / (max_score - expected)
                    homogeneity = max(0.0, min(1.0, homogeneity))
        
        stats.append(ColumnStats(
            position=pos,
            num_seqs=num_seqs,
            num_gaps=num_gaps,
            gap_fraction=gap_fraction,
            homogeneity=homogeneity,
            consensus=consensus
        ))
    
    return stats


def filter_alignment_columns(alignment: Alignment,
                              gap_threshold: float = 0.667,
                              homogeneity_threshold: float = 0.0) -> FilteredAlignment:
    """
    Filter alignment columns by quality criteria.
    
    Columns are kept if:
    - gap_fraction <= gap_threshold
    - homogeneity >= homogeneity_threshold
    
    Args:
        alignment: Input alignment
        gap_threshold: Maximum fraction of gaps (default: 0.667)
        homogeneity_threshold: Minimum homogeneity score (default: 0.0)
        
    Returns:
        FilteredAlignment with mapping information
    """
    column_stats = compute_column_stats(alignment)
    
    # Determine which positions pass filter
    filtered_positions = []
    full_to_filtered = {}
    filtered_to_full = {}
    
    filtered_idx = 0
    for i, stats in enumerate(column_stats):
        if stats.gap_fraction <= gap_threshold and stats.homogeneity >= homogeneity_threshold:
            filtered_positions.append(i)
            full_to_filtered[i] = filtered_idx
            filtered_to_full[filtered_idx] = i
            filtered_idx += 1
        else:
            full_to_filtered[i] = -1
    
    return FilteredAlignment(
        alignment=alignment,
        column_stats=column_stats,
        filtered_positions=filtered_positions,
        full_to_filtered=full_to_filtered,
        filtered_to_full=filtered_to_full
    )


def extract_filtered_sequences(filtered_ali: FilteredAlignment) -> Alignment:
    """
    Extract only the filtered (good) columns from an alignment.
    
    Args:
        filtered_ali: FilteredAlignment object
        
    Returns:
        New Alignment with only filtered columns
    """
    new_sequences = []
    
    for seq in filtered_ali.alignment.sequences:
        filtered_seq = ''.join(seq.sequence[pos] for pos in filtered_ali.filtered_positions)
        new_sequences.append(AlignedSequence(
            id=seq.id,
            sequence=filtered_seq,
            description=seq.description
        ))
    
    return Alignment(new_sequences)


def parse_hhalign_atab(atab_file: Union[str, Path]) -> List[Tuple[int, int]]:
    """
    Parse hhalign -atab output file.
    
    The atab file contains position correspondences between query and template.
    Format: lines with "pos1 pos2" where both > 0 indicates aligned positions.
    
    Args:
        atab_file: Path to .atab file from hhalign
        
    Returns:
        List of (query_pos, template_pos) tuples (1-based positions)
    """
    alignments = []
    
    with open(atab_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pos1 = int(parts[0])
                    pos2 = int(parts[1])
                    if pos1 > 0 and pos2 > 0:
                        alignments.append((pos1, pos2))
                except ValueError:
                    continue
    
    return alignments


def generate_consensus(alignment: Alignment,
                        gap_threshold: float = 1.0,
                        min_homogeneity: float = 0.0) -> str:
    """
    Generate consensus sequence from alignment.
    
    For each column:
    - If gap fraction > gap_threshold: output '-'
    - Else if homogeneity >= min_homogeneity: output best amino acid
    - Else: output 'x' (ambiguous)
    
    This replicates sr_filter -conplus logic with Yuri's defaults:
    -gcon=1 (gap_threshold=1.0) and -hcon=0 (min_homogeneity=0.0)
    
    Args:
        alignment: Input alignment
        gap_threshold: Max gap fraction for non-gap consensus (default: 1.0, matches Yuri)
        min_homogeneity: Min homogeneity for confident consensus (default: 0.0, matches Yuri)
        
    Returns:
        Consensus sequence string
    """
    column_stats = compute_column_stats(alignment)
    consensus = []
    
    for stats in column_stats:
        if stats.gap_fraction > gap_threshold:
            consensus.append('-')
        elif stats.homogeneity >= min_homogeneity:
            consensus.append(stats.consensus)
        else:
            consensus.append('x')
    
    return ''.join(consensus)


def add_consensus_to_alignment(alignment: Alignment,
                                consensus_name: str = "CONSENSUS") -> Alignment:
    """
    Add consensus sequence as first sequence in alignment.
    
    Args:
        alignment: Input alignment
        consensus_name: Name for consensus sequence
        
    Returns:
        New alignment with consensus as first sequence
    """
    consensus_seq = generate_consensus(alignment)
    
    consensus = AlignedSequence(
        id=consensus_name,
        sequence=consensus_seq,
        description=""
    )
    
    return Alignment([consensus] + list(alignment.sequences))


def run_hhalign_with_atab(query_fasta: Union[str, Path],
                          template_fasta: Union[str, Path],
                          output_dir: Union[str, Path],
                          global_alignment: bool = True) -> Tuple[Path, List[Tuple[int, int]]]:
    """
    Run hhalign and get alignment table.
    
    Args:
        query_fasta: Query MSA in FASTA format
        template_fasta: Template MSA in FASTA format
        output_dir: Directory for output files
        global_alignment: Use global alignment mode
        
    Returns:
        Tuple of (output_hhr_path, alignment_table)
    """
    query_fasta = Path(query_fasta)
    template_fasta = Path(template_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hhr_file = output_dir / "alignment.hhr"
    atab_file = output_dir / "alignment.atab"
    
    cmd = [
        'hhalign',
        '-i', str(query_fasta),
        '-t', str(template_fasta),
        '-o', str(hhr_file),
        '-atab', str(atab_file),
        '-id', '100',
        '-diff', 'inf',
        '-M', 'first',  # Use first sequence (consensus) as master
        '-v', '0'
    ]
    
    if global_alignment:
        cmd.append('-glob')
    else:
        cmd.append('-loc')
    
    logger.debug(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"hhalign failed: {result.stderr}")
    
    # Parse atab file
    alignments = parse_hhalign_atab(atab_file)
    
    return hhr_file, alignments


@dataclass
class MergeStats:
    """Statistics about the merge operation."""
    n_terminal_cols: int      # Columns in N-terminal tail
    core_aligned_cols: int    # Aligned core positions from hhalign
    insertion_cols: int       # Insertion columns between aligned positions
    c_terminal_cols: int      # Columns in C-terminal tail
    total_cols: int           # Total columns in merged alignment
    
    # Per-alignment breakdown
    ali1_n_tail: int          # N-tail columns from MSA1
    ali1_c_tail: int          # C-tail columns from MSA1
    ali2_n_tail: int          # N-tail columns from MSA2
    ali2_c_tail: int          # C-tail columns from MSA2


def merge_alignments(ali1: FilteredAlignment,
                     ali2: FilteredAlignment,
                     atab: List[Tuple[int, int]],
                     global_alignment: bool = True) -> Tuple[Alignment, MergeStats]:
    """
    Merge two alignments using alignment table from hhalign.
    
    This implements Yuri Wolf's algorithm for profile-profile merging:
    
    1. **N-terminal tail**: All columns BEFORE the first aligned position
       from BOTH alignments are included. If MSA1 has 5 columns before
       and MSA2 has 3, result has 5 columns (MSA2 gets gaps at start).
       
    2. **Core alignment**: Aligned positions from hhalign atab, plus
       ALL columns between consecutive aligned positions (including
       columns that were filtered out for hhalign). These "insertions"
       preserve the full alignment structure.
       
    3. **C-terminal tail**: All columns AFTER the last aligned position
       from BOTH alignments, same logic as N-terminal.
    
    Key insight: hhalign runs on FILTERED alignments (good columns only),
    but the merge reconstructs the FULL alignments including filtered
    columns as insertions. This preserves potentially important but
    gappy/variable regions.
    
    Args:
        ali1: First filtered alignment (contains both full and filtered data)
        ali2: Second filtered alignment
        atab: Alignment table (filtered_pos1, filtered_pos2) - 1-based!
        global_alignment: Include N/C terminal extensions
        
    Returns:
        Tuple of (Merged Alignment, MergeStats)
    """
    if not atab:
        raise ValueError("Empty alignment table - cannot merge alignments")
    
    # Convert atab to 0-based indices into filtered positions
    # atab contains 1-based positions in filtered alignment
    atab_0based = [(p1 - 1, p2 - 1) for p1, p2 in atab]
    
    # Get full alignment lengths
    full_len1 = len(ali1.column_stats)
    full_len2 = len(ali2.column_stats)
    
    num_seqs1 = len(ali1.alignment.sequences)
    num_seqs2 = len(ali2.alignment.sequences)
    
    # Build merged position list
    # Each entry is (pos_in_full1, pos_in_full2) where -1 means gap
    merged_positions = []
    
    # Statistics tracking
    n_terminal_cols = 0
    insertion_cols = 0
    c_terminal_cols = 0
    ali1_n_tail = 0
    ali2_n_tail = 0
    ali1_c_tail = 0
    ali2_c_tail = 0
    
    # === N-terminal extension (if global) ===
    # Include all columns BEFORE the first aligned position
    if global_alignment and atab_0based:
        first_filtered1, first_filtered2 = atab_0based[0]
        first_full1 = ali1.filtered_to_full.get(first_filtered1, 0)
        first_full2 = ali2.filtered_to_full.get(first_filtered2, 0)
        
        # Number of positions before first aligned position
        # This equals max(first_full1, first_full2) because:
        # - If first aligned is at position 4 (0-based), there are 4 columns before it (0,1,2,3)
        n_prefix = max(first_full1, first_full2)
        n_terminal_cols = n_prefix
        ali1_n_tail = first_full1
        ali2_n_tail = first_full2
        
        for i in range(n_prefix):
            # Calculate position relative to first aligned position
            pos1 = first_full1 - n_prefix + i
            pos2 = first_full2 - n_prefix + i
            merged_positions.append((
                pos1 if pos1 >= 0 else -1,
                pos2 if pos2 >= 0 else -1
            ))
        
        if n_prefix > 0:
            logger.debug(f"  N-terminal tail: {n_prefix} cols (MSA1: {first_full1}, MSA2: {first_full2})")
    
    # === Body - process aligned positions ===
    prev_full1 = -1
    prev_full2 = -1
    core_positions = 0
    
    for idx, (filt1, filt2) in enumerate(atab_0based):
        # Get full positions via mapping
        # filt1/filt2 are positions in FILTERED alignment
        # filtered_to_full maps them to positions in FULL alignment
        curr_full1 = ali1.filtered_to_full.get(filt1)
        curr_full2 = ali2.filtered_to_full.get(filt2)
        
        if curr_full1 is None or curr_full2 is None:
            logger.warning(f"  Missing mapping for filtered positions ({filt1}, {filt2})")
            continue
        
        # Handle insertions between previous and current aligned positions
        # This includes ALL columns (even filtered ones) in the FULL alignment
        if prev_full1 >= 0:
            # Insertions = all positions between prev+1 and curr-1 inclusive
            ins1 = curr_full1 - prev_full1 - 1
            ins2 = curr_full2 - prev_full2 - 1
            
            # Take maximum insertions, fill with gaps where one alignment is shorter
            max_ins = max(ins1, ins2)
            insertion_cols += max_ins
            
            for j in range(max_ins):
                p1 = prev_full1 + 1 + j if j < ins1 else -1
                p2 = prev_full2 + 1 + j if j < ins2 else -1
                merged_positions.append((p1, p2))
        
        # Add the aligned position itself
        merged_positions.append((curr_full1, curr_full2))
        core_positions += 1
        
        prev_full1 = curr_full1
        prev_full2 = curr_full2
    
    # === C-terminal extension (if global) ===
    # Include all columns AFTER the last aligned position
    if global_alignment and prev_full1 >= 0:
        # Positions after last aligned position
        remaining1 = full_len1 - prev_full1 - 1
        remaining2 = full_len2 - prev_full2 - 1
        
        max_remaining = max(remaining1, remaining2)
        c_terminal_cols = max_remaining
        ali1_c_tail = remaining1
        ali2_c_tail = remaining2
        
        for i in range(max_remaining):
            p1 = prev_full1 + 1 + i if i < remaining1 else -1
            p2 = prev_full2 + 1 + i if i < remaining2 else -1
            merged_positions.append((p1, p2))
        
        if max_remaining > 0:
            logger.debug(f"  C-terminal tail: {max_remaining} cols (MSA1: {remaining1}, MSA2: {remaining2})")
    
    # Log merge composition
    total_cols = len(merged_positions)
    logger.debug(f"  Merge composition: N-tail={n_terminal_cols}, core={core_positions}, "
                 f"insertions={insertion_cols}, C-tail={c_terminal_cols}, total={total_cols}")
    
    # === Build merged sequences ===
    merged_seqs = []
    
    # Add sequences from alignment 1
    for seq in ali1.alignment.sequences:
        merged_seq = ''
        for pos1, pos2 in merged_positions:
            if pos1 >= 0 and pos1 < len(seq.sequence):
                merged_seq += seq.sequence[pos1]
            else:
                merged_seq += '-'
        merged_seqs.append(AlignedSequence(
            id=seq.id,
            sequence=merged_seq,
            description=seq.description
        ))
    
    # Add sequences from alignment 2
    for seq in ali2.alignment.sequences:
        merged_seq = ''
        for pos1, pos2 in merged_positions:
            if pos2 >= 0 and pos2 < len(seq.sequence):
                merged_seq += seq.sequence[pos2]
            else:
                merged_seq += '-'
        merged_seqs.append(AlignedSequence(
            id=seq.id,
            sequence=merged_seq,
            description=seq.description
        ))
    
    # === Validation ===
    # All sequences must have the same length
    if merged_seqs:
        expected_len = len(merged_seqs[0].sequence)
        for seq in merged_seqs:
            if len(seq.sequence) != expected_len:
                raise RuntimeError(
                    f"Merge validation failed: sequence {seq.id} has length "
                    f"{len(seq.sequence)}, expected {expected_len}"
                )
    
    stats = MergeStats(
        n_terminal_cols=n_terminal_cols,
        core_aligned_cols=core_positions,
        insertion_cols=insertion_cols,
        c_terminal_cols=c_terminal_cols,
        total_cols=total_cols,
        ali1_n_tail=ali1_n_tail,
        ali1_c_tail=ali1_c_tail,
        ali2_n_tail=ali2_n_tail,
        ali2_c_tail=ali2_c_tail
    )
    
    return Alignment(merged_seqs), stats


def merge_two_msas(msa1_path: Union[str, Path],
                   msa2_path: Union[str, Path],
                   output_path: Union[str, Path] = None,
                   gap_threshold: float = 0.667,
                   homogeneity_threshold: float = 0.0,
                   global_alignment: bool = True,
                   work_dir: Union[str, Path] = None) -> Alignment:
    """
    Merge two MSA files using profile-guided alignment.
    
    This is the main entry point for Yuri's merging algorithm.
    
    Args:
        msa1_path: Path to first MSA (FASTA)
        msa2_path: Path to second MSA (FASTA)
        output_path: Optional path to save merged alignment
        gap_threshold: Max gap fraction for column filtering (default: 0.667)
        homogeneity_threshold: Min homogeneity for column filtering (default: 0.0)
        global_alignment: Use global alignment mode (default: True)
        work_dir: Working directory for temp files (default: auto)
        
    Returns:
        Merged Alignment object
    """
    msa1_path = Path(msa1_path)
    msa2_path = Path(msa2_path)
    
    # Read input alignments
    logger.info(f"Reading {msa1_path.name} and {msa2_path.name}")
    ali1 = read_fasta(msa1_path)
    ali2 = read_fasta(msa2_path)
    
    logger.info(f"  MSA1: {len(ali1)} seqs, {ali1.get_length()} cols")
    logger.info(f"  MSA2: {len(ali2)} seqs, {ali2.get_length()} cols")
    
    # Filter columns
    logger.info(f"Filtering columns (gap<={gap_threshold}, hom>={homogeneity_threshold})")
    filtered1 = filter_alignment_columns(ali1, gap_threshold, homogeneity_threshold)
    filtered2 = filter_alignment_columns(ali2, gap_threshold, homogeneity_threshold)
    
    logger.info(f"  MSA1: {len(filtered1.filtered_positions)}/{ali1.get_length()} cols kept")
    logger.info(f"  MSA2: {len(filtered2.filtered_positions)}/{ali2.get_length()} cols kept")
    
    # Create working directory
    if work_dir:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_context = None
    else:
        tmp_context = tempfile.TemporaryDirectory()
        work_dir = Path(tmp_context.name)
    
    try:
        # Extract filtered sequences
        filtered_seqs1 = extract_filtered_sequences(filtered1)
        filtered_seqs2 = extract_filtered_sequences(filtered2)
        
        # Add consensus as first sequence (critical for -M first to work properly!)
        # This ensures hhalign builds profile from consensus, not from potentially
        # fragmented first sequence
        name1 = msa1_path.stem
        name2 = msa2_path.stem
        
        filtered_with_cons1 = add_consensus_to_alignment(filtered_seqs1, f"{name1}@_cons_1")
        filtered_with_cons2 = add_consensus_to_alignment(filtered_seqs2, f"{name2}@_cons_2")
        
        logger.info(f"  Added consensus sequences for hhalign")
        
        # Write filtered alignments with consensus for hhalign
        filtered_fasta1 = work_dir / "filtered1.fasta"
        filtered_fasta2 = work_dir / "filtered2.fasta"
        
        write_fasta(filtered_with_cons1, filtered_fasta1)
        write_fasta(filtered_with_cons2, filtered_fasta2)
        
        # Run hhalign on filtered alignments
        logger.info("Running hhalign on filtered alignments")
        hhr_file, atab = run_hhalign_with_atab(
            filtered_fasta1, 
            filtered_fasta2,
            work_dir,
            global_alignment=global_alignment
        )
        
        logger.info(f"  Got {len(atab)} aligned position pairs")
        
        if not atab:
            raise RuntimeError("hhalign produced no alignments")
        
        # Merge alignments (using original filtered alignments without consensus)
        logger.info("Merging alignments")
        merged, stats = merge_alignments(
            filtered1, 
            filtered2, 
            atab,
            global_alignment=global_alignment
        )
        
        logger.info(f"  Merged: {len(merged)} seqs, {merged.get_length()} cols")
        logger.info(f"  Composition: N-tail={stats.n_terminal_cols}, core={stats.core_aligned_cols}, "
                    f"insertions={stats.insertion_cols}, C-tail={stats.c_terminal_cols}")
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            write_fasta(merged, output_path)
            logger.info(f"  Saved to {output_path}")
        
        return merged
        
    finally:
        if tmp_context:
            tmp_context.cleanup()


