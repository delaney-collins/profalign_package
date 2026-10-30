#!/usr/bin/env python3
"""
Test merge algorithm with focus on N/C terminal tail handling.

This test verifies that our implementation matches Yuri Wolf's algorithm
for profile-profile merging, specifically:

1. N-terminal tail: All columns BEFORE first aligned position
2. Core: Aligned positions + insertions between them
3. C-terminal tail: All columns AFTER last aligned position

The key insight is that hhalign runs on FILTERED alignments (good columns only),
but the merge reconstructs FULL alignments including filtered columns.
"""

from pathlib import Path

from merge import (
    Alignment, AlignedSequence, FilteredAlignment, ColumnStats,
    merge_alignments, filter_alignment_columns, MergeStats
)


def make_alignment(seqs: dict) -> Alignment:
    """Create alignment from dict of {id: sequence}."""
    return Alignment([
        AlignedSequence(id=name, sequence=seq, description="")
        for name, seq in seqs.items()
    ])


def make_filtered_alignment(alignment: Alignment, 
                            good_positions: list) -> FilteredAlignment:
    """
    Create FilteredAlignment with specified good (filtered) positions.
    
    Args:
        alignment: Full alignment
        good_positions: List of 0-based positions that pass filter
    """
    ali_len = alignment.get_length()
    
    # Create column stats (simplified)
    stats = []
    for i in range(ali_len):
        is_good = i in good_positions
        stats.append(ColumnStats(
            position=i,
            num_seqs=len(alignment),
            num_gaps=0 if is_good else 5,  # Fake: good=no gaps, bad=many gaps
            gap_fraction=0.0 if is_good else 0.8,
            homogeneity=1.0 if is_good else 0.0,
            consensus='A'
        ))
    
    # Build mappings
    full_to_filtered = {}
    filtered_to_full = {}
    filtered_idx = 0
    for full_idx in range(ali_len):
        if full_idx in good_positions:
            full_to_filtered[full_idx] = filtered_idx
            filtered_to_full[filtered_idx] = full_idx
            filtered_idx += 1
        else:
            full_to_filtered[full_idx] = -1
    
    return FilteredAlignment(
        alignment=alignment,
        column_stats=stats,
        filtered_positions=good_positions,
        full_to_filtered=full_to_filtered,
        filtered_to_full=filtered_to_full
    )


def test_basic_merge_no_tails():
    """Test merge where first/last positions are aligned (no tails)."""
    print("=" * 60)
    print("TEST: Basic merge with no tails")
    print("=" * 60)
    
    # MSA1: 5 columns, all good
    # Positions: 0 1 2 3 4
    ali1 = make_alignment({
        'seq1a': 'ACDEF',
        'seq1b': 'ACDEY',
    })
    filtered1 = make_filtered_alignment(ali1, [0, 1, 2, 3, 4])
    
    # MSA2: 5 columns, all good
    ali2 = make_alignment({
        'seq2a': 'GHIKL',
        'seq2b': 'GHIKM',
    })
    filtered2 = make_filtered_alignment(ali2, [0, 1, 2, 3, 4])
    
    # atab: all positions align 1:1 (1-based)
    atab = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    
    merged, stats = merge_alignments(filtered1, filtered2, atab)
    
    print(f"MSA1: {ali1.get_length()} cols, filtered: {filtered1.filtered_positions}")
    print(f"MSA2: {ali2.get_length()} cols, filtered: {filtered2.filtered_positions}")
    print(f"Merged: {merged.get_length()} cols")
    print(f"Stats: N-tail={stats.n_terminal_cols}, core={stats.core_aligned_cols}, "
          f"ins={stats.insertion_cols}, C-tail={stats.c_terminal_cols}")
    
    for seq in merged.sequences:
        print(f"  {seq.id}: {seq.sequence}")
    
    assert merged.get_length() == 5, f"Expected 5 cols, got {merged.get_length()}"
    assert stats.n_terminal_cols == 0
    assert stats.c_terminal_cols == 0
    assert stats.core_aligned_cols == 5
    print("✓ PASSED\n")


def test_n_terminal_tail():
    """Test merge with N-terminal tail."""
    print("=" * 60)
    print("TEST: N-terminal tail")
    print("=" * 60)
    
    # MSA1: 7 columns
    # Full:     0  1  2  3  4  5  6
    #           N  N  N  C  C  C  C   (N=N-tail, C=core)
    # Filtered:          0  1  2  3
    ali1 = make_alignment({
        'seq1a': 'AAACDEF',  # First 3 are N-tail
        'seq1b': 'BBBCDEY',
    })
    filtered1 = make_filtered_alignment(ali1, [3, 4, 5, 6])
    
    # MSA2: 5 columns
    # Full:     0  1  2  3  4
    #           N  C  C  C  C   (N=N-tail, C=core)
    # Filtered: 0  1  2  3  4   (all pass except position 0 is N-tail context)
    # Wait, let me re-think: filtered positions determine what hhalign sees
    # If filtered = [1,2,3,4], first aligned is at full pos 1
    ali2 = make_alignment({
        'seq2a': 'XHIKL',  # First 1 is N-tail
        'seq2b': 'YHIKM',
    })
    filtered2 = make_filtered_alignment(ali2, [1, 2, 3, 4])
    
    # atab: filtered positions align
    # MSA1 filtered [0,1,2,3] ↔ MSA2 filtered [0,1,2,3]
    # Full: MSA1 [3,4,5,6] ↔ MSA2 [1,2,3,4]
    atab = [(1, 1), (2, 2), (3, 3), (4, 4)]
    
    merged, stats = merge_alignments(filtered1, filtered2, atab)
    
    print(f"MSA1: {ali1.get_length()} cols, filtered: {filtered1.filtered_positions}")
    print(f"  First aligned at full pos: {filtered1.filtered_to_full[0]}")
    print(f"MSA2: {ali2.get_length()} cols, filtered: {filtered2.filtered_positions}")
    print(f"  First aligned at full pos: {filtered2.filtered_to_full[0]}")
    print(f"Merged: {merged.get_length()} cols")
    print(f"Stats: N-tail={stats.n_terminal_cols}, core={stats.core_aligned_cols}, "
          f"ins={stats.insertion_cols}, C-tail={stats.c_terminal_cols}")
    
    for seq in merged.sequences:
        print(f"  {seq.id}: {seq.sequence}")
    
    # MSA1 first aligned at full pos 3 (0-based)
    # MSA2 first aligned at full pos 1 (0-based)
    # N-tail = max(3, 1) = 3 columns
    # Core = 4 aligned positions
    # Total = 3 + 4 = 7
    
    assert stats.n_terminal_cols == 3, f"Expected N-tail=3, got {stats.n_terminal_cols}"
    assert stats.core_aligned_cols == 4, f"Expected core=4, got {stats.core_aligned_cols}"
    assert merged.get_length() == 7, f"Expected 7 cols, got {merged.get_length()}"
    
    # Verify sequences
    # MSA1 seqs should have: full positions 0,1,2,3,4,5,6 (all 7)
    # MSA2 seqs should have: gaps for 2 positions, then full 0,1,2,3,4
    seq2a = [s for s in merged.sequences if s.id == 'seq2a'][0]
    assert seq2a.sequence[:2] == '--', f"Expected gaps at start, got {seq2a.sequence[:2]}"
    assert seq2a.sequence[2:] == 'XHIKL', f"Expected XHIKL, got {seq2a.sequence[2:]}"
    
    print("✓ PASSED\n")


def test_c_terminal_tail():
    """Test merge with C-terminal tail."""
    print("=" * 60)
    print("TEST: C-terminal tail")
    print("=" * 60)
    
    # MSA1: 5 columns
    # Full:     0  1  2  3  4
    # Filtered: 0  1  2       (positions 3,4 are C-tail)
    ali1 = make_alignment({
        'seq1a': 'CDEXX',
        'seq1b': 'CDEYY',
    })
    filtered1 = make_filtered_alignment(ali1, [0, 1, 2])
    
    # MSA2: 7 columns
    # Full:     0  1  2  3  4  5  6
    # Filtered: 0  1  2              (positions 3-6 are C-tail)
    ali2 = make_alignment({
        'seq2a': 'HIKMMMM',
        'seq2b': 'HIKNNNN',
    })
    filtered2 = make_filtered_alignment(ali2, [0, 1, 2])
    
    # atab: 3 aligned positions
    atab = [(1, 1), (2, 2), (3, 3)]
    
    merged, stats = merge_alignments(filtered1, filtered2, atab)
    
    print(f"MSA1: {ali1.get_length()} cols, filtered: {filtered1.filtered_positions}")
    print(f"MSA2: {ali2.get_length()} cols, filtered: {filtered2.filtered_positions}")
    print(f"Merged: {merged.get_length()} cols")
    print(f"Stats: N-tail={stats.n_terminal_cols}, core={stats.core_aligned_cols}, "
          f"ins={stats.insertion_cols}, C-tail={stats.c_terminal_cols}")
    
    for seq in merged.sequences:
        print(f"  {seq.id}: {seq.sequence}")
    
    # Last aligned at:
    # MSA1: full pos 2, remaining = 5-2-1 = 2 (positions 3,4)
    # MSA2: full pos 2, remaining = 7-2-1 = 4 (positions 3,4,5,6)
    # C-tail = max(2, 4) = 4
    
    assert stats.c_terminal_cols == 4, f"Expected C-tail=4, got {stats.c_terminal_cols}"
    assert stats.core_aligned_cols == 3
    assert merged.get_length() == 7, f"Expected 7 cols, got {merged.get_length()}"
    
    # Verify MSA1 seqs have gaps at end
    seq1a = [s for s in merged.sequences if s.id == 'seq1a'][0]
    assert seq1a.sequence == 'CDEXX--', f"Expected CDEXX--, got {seq1a.sequence}"
    
    print("✓ PASSED\n")


def test_insertions_between_aligned():
    """Test merge with insertions (filtered-out columns) between aligned positions."""
    print("=" * 60)
    print("TEST: Insertions between aligned positions")
    print("=" * 60)
    
    # MSA1: 8 columns
    # Full:     0  1  2  3  4  5  6  7
    # Filtered: 0        1        2  3   (positions 1,2,4,5 are insertions)
    ali1 = make_alignment({
        'seq1a': 'AxxBxxCD',
        'seq1b': 'AyyByyCD',
    })
    filtered1 = make_filtered_alignment(ali1, [0, 3, 6, 7])
    
    # MSA2: 6 columns
    # Full:     0  1  2  3  4  5
    # Filtered: 0     1     2  3   (positions 1,3 are insertions)
    ali2 = make_alignment({
        'seq2a': 'EzFzGH',
        'seq2b': 'EwFwGH',
    })
    filtered2 = make_filtered_alignment(ali2, [0, 2, 4, 5])
    
    # atab: 4 aligned positions (filtered indices)
    atab = [(1, 1), (2, 2), (3, 3), (4, 4)]
    
    merged, stats = merge_alignments(filtered1, filtered2, atab)
    
    print(f"MSA1: {ali1.get_length()} cols, filtered: {filtered1.filtered_positions}")
    print(f"  filtered_to_full: {filtered1.filtered_to_full}")
    print(f"MSA2: {ali2.get_length()} cols, filtered: {filtered2.filtered_positions}")
    print(f"  filtered_to_full: {filtered2.filtered_to_full}")
    print(f"Merged: {merged.get_length()} cols")
    print(f"Stats: N-tail={stats.n_terminal_cols}, core={stats.core_aligned_cols}, "
          f"ins={stats.insertion_cols}, C-tail={stats.c_terminal_cols}")
    
    for seq in merged.sequences:
        print(f"  {seq.id}: {seq.sequence}")
    
    # Between aligned positions:
    # After pos 0: MSA1 has 2 ins (1,2), MSA2 has 1 ins (1) → max=2
    # After pos 1: MSA1 has 2 ins (4,5), MSA2 has 1 ins (3) → max=2
    # After pos 2: MSA1 has 0 ins, MSA2 has 0 ins → max=0
    # Total insertions = 2 + 2 + 0 = 4
    
    assert stats.insertion_cols == 4, f"Expected ins=4, got {stats.insertion_cols}"
    assert stats.core_aligned_cols == 4
    assert merged.get_length() == 8, f"Expected 8 cols, got {merged.get_length()}"
    
    print("✓ PASSED\n")


def test_full_scenario():
    """Test realistic scenario with all components: N-tail, insertions, C-tail."""
    print("=" * 60)
    print("TEST: Full scenario (N-tail + insertions + C-tail)")
    print("=" * 60)
    
    # MSA1: 12 columns
    # Full:     0  1  2  3  4  5  6  7  8  9 10 11
    #           N  N  C     C     C  C  C        C-tail
    # Filtered:       0     1     2  3  4        (5 filtered positions)
    # N-tail: 0,1 (2 cols), C-tail: 9,10,11 (3 cols)
    # Insertions: between 2-4 (pos 3), between 4-6 (pos 5)
    ali1 = make_alignment({
        'seq1a': 'NNAxBxCDEzzz',
        'seq1b': 'NNAyByCD-zzz',
    })
    filtered1 = make_filtered_alignment(ali1, [2, 4, 6, 7, 8])
    
    # MSA2: 10 columns
    # Full:     0  1  2  3  4  5  6  7  8  9
    #           N  C     C     C  C  C  T  T
    # Filtered:    0     1     2  3  4        (5 filtered positions)
    # N-tail: 0 (1 col), C-tail: 8,9 (2 cols)
    ali2 = make_alignment({
        'seq2a': 'nFwGwHIJtt',
        'seq2b': 'nFvGvHIJtt',
    })
    filtered2 = make_filtered_alignment(ali2, [1, 3, 5, 6, 7])
    
    # atab: 5 aligned positions
    atab = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    
    merged, stats = merge_alignments(filtered1, filtered2, atab)
    
    print(f"MSA1: {ali1.get_length()} cols")
    print(f"  filtered_to_full: {filtered1.filtered_to_full}")
    print(f"  First aligned full pos: {filtered1.filtered_to_full[0]}")
    print(f"  Last aligned full pos: {filtered1.filtered_to_full[4]}")
    print(f"MSA2: {ali2.get_length()} cols")
    print(f"  filtered_to_full: {filtered2.filtered_to_full}")
    print(f"  First aligned full pos: {filtered2.filtered_to_full[0]}")
    print(f"  Last aligned full pos: {filtered2.filtered_to_full[4]}")
    print(f"Merged: {merged.get_length()} cols")
    print(f"Stats: N-tail={stats.n_terminal_cols}, core={stats.core_aligned_cols}, "
          f"ins={stats.insertion_cols}, C-tail={stats.c_terminal_cols}")
    print(f"  ali1: N-tail={stats.ali1_n_tail}, C-tail={stats.ali1_c_tail}")
    print(f"  ali2: N-tail={stats.ali2_n_tail}, C-tail={stats.ali2_c_tail}")
    
    for seq in merged.sequences:
        print(f"  {seq.id}: {seq.sequence}")
    
    # N-terminal:
    # MSA1 first aligned at full pos 2, MSA2 at full pos 1
    # N-tail = max(2, 1) = 2
    
    # C-terminal:
    # MSA1 last aligned at full pos 8, remaining = 12-8-1 = 3
    # MSA2 last aligned at full pos 7, remaining = 10-7-1 = 2
    # C-tail = max(3, 2) = 3
    
    # Insertions:
    # Between aligned positions (full positions):
    # MSA1: 2→4 (1 ins at 3), 4→6 (1 ins at 5), 6→7 (0), 7→8 (0) = 2 total
    # MSA2: 1→3 (1 ins at 2), 3→5 (1 ins at 4), 5→6 (0), 6→7 (0) = 2 total
    # Per gap: max(1,1)=1, max(1,1)=1, max(0,0)=0, max(0,0)=0 = 2 total
    
    assert stats.n_terminal_cols == 2, f"Expected N-tail=2, got {stats.n_terminal_cols}"
    assert stats.c_terminal_cols == 3, f"Expected C-tail=3, got {stats.c_terminal_cols}"
    assert stats.core_aligned_cols == 5, f"Expected core=5, got {stats.core_aligned_cols}"
    assert stats.insertion_cols == 2, f"Expected ins=2, got {stats.insertion_cols}"
    
    # Total = N-tail(2) + ins(2) + core(5) + C-tail(3) = 12
    assert merged.get_length() == 12, f"Expected 12 cols, got {merged.get_length()}"
    
    print("✓ PASSED\n")


def test_sequence_content_preserved():
    """Verify that original sequence content is correctly preserved."""
    print("=" * 60)
    print("TEST: Sequence content preservation")
    print("=" * 60)
    
    # Simple case: one sequence per MSA, asymmetric tails
    ali1 = make_alignment({'s1': 'AAACCC'})  # 6 cols
    filtered1 = make_filtered_alignment(ali1, [2, 3, 4])  # First aligned at pos 2
    
    ali2 = make_alignment({'s2': 'XYYY'})  # 4 cols
    filtered2 = make_filtered_alignment(ali2, [0, 1, 2])  # First aligned at pos 0
    
    # atab: 3 aligned positions
    atab = [(1, 1), (2, 2), (3, 3)]
    
    merged, stats = merge_alignments(filtered1, filtered2, atab)
    
    print(f"MSA1: '{ali1.sequences[0].sequence}' ({ali1.get_length()} cols)")
    print(f"  Filtered positions: {filtered1.filtered_positions}")
    print(f"MSA2: '{ali2.sequences[0].sequence}' ({ali2.get_length()} cols)")
    print(f"  Filtered positions: {filtered2.filtered_positions}")
    
    for seq in merged.sequences:
        print(f"  {seq.id}: {seq.sequence}")
    
    # Expected:
    # N-tail = max(2, 0) = 2 cols → s1 gets AA, s2 gets --
    # Core = 3 aligned + 0 insertions
    # C-tail = max(6-4-1=1, 4-2-1=1) = 1 col → s1 gets C, s2 gets Y
    # Total = 2 + 3 + 1 = 6
    
    s1 = [s for s in merged.sequences if s.id == 's1'][0]
    s2 = [s for s in merged.sequences if s.id == 's2'][0]
    
    assert s1.sequence == 'AAACCC', f"s1 should be AAACCC, got {s1.sequence}"
    assert s2.sequence == '--XYYY', f"s2 should be --XYYY, got {s2.sequence}"
    
    # Verify no residues are lost
    s1_residues = s1.sequence.replace('-', '')
    s2_residues = s2.sequence.replace('-', '')
    assert s1_residues == 'AAACCC', f"s1 residues: expected AAACCC, got {s1_residues}"
    assert s2_residues == 'XYYY', f"s2 residues: expected XYYY, got {s2_residues}"
    
    print("✓ PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MERGE ALGORITHM TESTS - N/C TERMINAL TAILS")
    print("=" * 60 + "\n")
    
    test_basic_merge_no_tails()
    test_n_terminal_tail()
    test_c_terminal_tail()
    test_insertions_between_aligned()
    test_full_scenario()
    test_sequence_content_preserved()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
