"""
formats.py - Reading and writing alignment formats (FASTA, seqrows)

Seqrows format is a simple tab-separated format:
    ID<TAB>ALIGNED_SEQUENCE

This module provides conversion between FASTA and seqrows formats.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class AlignedSequence:
    """Single sequence from an alignment."""
    id: str
    sequence: str
    description: str = ""


class Alignment:
    """
    Multiple sequence alignment container.
    
    Stores sequences and provides conversion between formats.
    """
    
    def __init__(self, sequences: List[AlignedSequence] = None):
        self.sequences = sequences or []
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __iter__(self):
        return iter(self.sequences)
    
    def add(self, seq: AlignedSequence):
        """Add a sequence to the alignment."""
        self.sequences.append(seq)
    
    def get_ids(self) -> List[str]:
        """Return list of sequence IDs."""
        return [s.id for s in self.sequences]
    
    def get_length(self) -> int:
        """Return alignment length (number of columns)."""
        if not self.sequences:
            return 0
        return len(self.sequences[0].sequence)
    
    def to_dict(self) -> Dict[str, str]:
        """Return alignment as {id: sequence} dictionary."""
        return {s.id: s.sequence for s in self.sequences}


def read_fasta(filepath: Union[str, Path]) -> Alignment:
    """
    Read FASTA file (aligned or unaligned).
    
    Args:
        filepath: Path to FASTA file
        
    Returns:
        Alignment object containing all sequences
    """
    filepath = Path(filepath)
    alignment = Alignment()
    
    current_id = None
    current_desc = ""
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n\r')
            
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    alignment.add(AlignedSequence(
                        id=current_id,
                        sequence=''.join(current_seq),
                        description=current_desc
                    ))
                
                # Parse header
                header = line[1:].strip()
                parts = header.split(None, 1)  # Split on first whitespace
                current_id = parts[0] if parts else ""
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq = []
            else:
                # Sequence line - remove whitespace
                current_seq.append(line.replace(' ', '').replace('\t', ''))
    
    # Don't forget last sequence
    if current_id is not None:
        alignment.add(AlignedSequence(
            id=current_id,
            sequence=''.join(current_seq),
            description=current_desc
        ))
    
    return alignment


def write_fasta(alignment: Alignment, filepath: Union[str, Path], 
                line_width: int = 80, include_description: bool = False):
    """
    Write alignment to FASTA file.
    
    Args:
        alignment: Alignment object to write
        filepath: Output file path
        line_width: Maximum line width for sequence (0 = no wrapping)
        include_description: Include description in header
    """
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        for seq in alignment:
            # Write header
            if include_description and seq.description:
                f.write(f">{seq.id} {seq.description}\n")
            else:
                f.write(f">{seq.id}\n")
            
            # Write sequence (with optional line wrapping)
            if line_width > 0:
                for i in range(0, len(seq.sequence), line_width):
                    f.write(seq.sequence[i:i+line_width] + '\n')
            else:
                f.write(seq.sequence + '\n')


def read_seqrows(filepath: Union[str, Path]) -> Alignment:
    """
    Read seqrows format file.
    
    Seqrows format: ID<TAB>SEQUENCE (one per line)
    
    Args:
        filepath: Path to seqrows file
        
    Returns:
        Alignment object
    """
    filepath = Path(filepath)
    alignment = Alignment()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t', 1)
            if len(parts) >= 2:
                seq_id = parts[0].strip()
                sequence = parts[1].strip()
                alignment.add(AlignedSequence(id=seq_id, sequence=sequence))
            elif len(parts) == 1:
                # Handle space-separated format as fallback
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    seq_id = parts[0].strip()
                    sequence = parts[1].strip()
                    alignment.add(AlignedSequence(id=seq_id, sequence=sequence))
    
    return alignment


def write_seqrows(alignment: Alignment, filepath: Union[str, Path]):
    """
    Write alignment to seqrows format.
    
    Args:
        alignment: Alignment object to write
        filepath: Output file path
    """
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        for seq in alignment:
            f.write(f"{seq.id}\t{seq.sequence}\n")


def get_alignment_stats(alignment: Alignment) -> Dict:
    """
    Calculate basic alignment statistics.
    
    Args:
        alignment: Alignment object
        
    Returns:
        Dictionary with statistics
    """
    if not alignment.sequences:
        return {
            'num_sequences': 0,
            'alignment_length': 0,
            'min_seq_length': 0,
            'max_seq_length': 0,
        }
    
    lengths = [len(s.sequence.replace('-', '')) for s in alignment]
    
    return {
        'num_sequences': len(alignment),
        'alignment_length': alignment.get_length(),
        'min_seq_length': min(lengths),
        'max_seq_length': max(lengths),
        'avg_seq_length': sum(lengths) / len(lengths),
    }
