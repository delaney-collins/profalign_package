"""
provenance.py - Track sequence origins through the merge pipeline

This module provides utilities to:
1. Generate .cls files (profile -> sequence ID mapping)
2. Trace merge history to find original profiles
3. Build a complete provenance tree

Equivalent to Yuri's .cls file generation at the end of prof_align
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import sys
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))

from formats import read_fasta

logger = logging.getLogger(__name__)


@dataclass
class ProfileInfo:
    """Information about a profile and its sequences."""
    name: str
    path: Path
    sequence_ids: List[str]
    
    @property
    def count(self) -> int:
        return len(self.sequence_ids)


def extract_sequence_ids(profile_path: Path) -> List[str]:
    """
    Extract all sequence IDs from a profile (FASTA file).
    
    Args:
        profile_path: Path to FASTA file
        
    Returns:
        List of sequence IDs (first word of each header)
    """
    alignment = read_fasta(profile_path)
    return [seq.id for seq in alignment.sequences]


def generate_cls_file(profiles_dir: Path, output_path: Path) -> Dict[str, List[str]]:
    """
    Generate a .cls file mapping profiles to their sequences.
    
    This replicates Yuri's $name.cls output format:
        <count> <profile_name>\t<seq1> <seq2> <seq3>...
    
    Args:
        profiles_dir: Directory containing final profile FASTA files
        output_path: Path to write .cls file
        
    Returns:
        Dict mapping profile name -> list of sequence IDs
    """
    profile_map: Dict[str, List[str]] = {}
    
    # Find all FASTA files
    fasta_files = list(profiles_dir.glob("*.fasta")) + list(profiles_dir.glob("*.fa"))
    
    with open(output_path, 'w') as f:
        for fasta_path in sorted(fasta_files):
            profile_name = fasta_path.stem
            seq_ids = extract_sequence_ids(fasta_path)
            profile_map[profile_name] = seq_ids
            
            # Write in Yuri's format
            f.write(f"{len(seq_ids)} {profile_name}\t{' '.join(seq_ids)}\n")
    
    logger.info(f"Generated .cls file with {len(profile_map)} profiles")
    return profile_map


@dataclass
class MergeNode:
    """A node in the merge history tree."""
    name: str
    parent1: Optional[str] = None
    parent2: Optional[str] = None
    iteration: int = 0
    is_original: bool = True
    
    @property
    def is_merge(self) -> bool:
        return self.parent1 is not None


def load_merge_history(history_path: Path) -> Dict[str, MergeNode]:
    """
    Load merge history from TSV file.
    
    Args:
        history_path: Path to merge_history.tsv
        
    Returns:
        Dict mapping merge result ID -> MergeNode
    """
    nodes: Dict[str, MergeNode] = {}
    
    with open(history_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                result_id, parent1, parent2 = parts[:3]
                iteration = int(parts[3]) if len(parts) > 3 else 0
                nodes[result_id] = MergeNode(
                    name=result_id,
                    parent1=parent1,
                    parent2=parent2,
                    iteration=iteration,
                    is_original=False
                )
    
    return nodes


def trace_origins(profile_name: str, merge_history: Dict[str, MergeNode]) -> Set[str]:
    """
    Trace a profile back to its original (unmerged) profiles.
    
    Args:
        profile_name: Name of profile to trace
        merge_history: Dict from load_merge_history()
        
    Returns:
        Set of original profile names that were merged into this one
    """
    origins: Set[str] = set()
    
    def trace_recursive(name: str):
        if name in merge_history:
            node = merge_history[name]
            trace_recursive(node.parent1)
            trace_recursive(node.parent2)
        else:
            # This is an original profile
            origins.add(name)
    
    trace_recursive(profile_name)
    return origins


def build_merge_tree_text(profile_name: str, merge_history: Dict[str, MergeNode], 
                          indent: int = 0) -> str:
    """
    Build a text representation of the merge tree.
    
    Args:
        profile_name: Name of profile to show tree for
        merge_history: Dict from load_merge_history()
        indent: Current indentation level
        
    Returns:
        Multi-line string showing merge tree
    """
    prefix = "  " * indent
    
    if profile_name in merge_history:
        node = merge_history[profile_name]
        lines = [f"{prefix}{profile_name} (merged, iter={node.iteration})"]
        lines.append(build_merge_tree_text(node.parent1, merge_history, indent + 1))
        lines.append(build_merge_tree_text(node.parent2, merge_history, indent + 1))
        return '\n'.join(lines)
    else:
        return f"{prefix}{profile_name} (original)"


def generate_full_provenance(profiles_dir: Path, 
                             merge_history_path: Path,
                             output_path: Path) -> None:
    """
    Generate a complete provenance report.
    
    For each final profile, shows:
    - All sequence IDs it contains
    - Which original profiles it came from
    - The merge tree
    
    Args:
        profiles_dir: Directory with final profiles
        merge_history_path: Path to merge_history.tsv
        output_path: Path to write provenance report
    """
    # Load merge history
    merge_history = load_merge_history(merge_history_path)
    
    # Find all profiles
    fasta_files = list(profiles_dir.glob("*.fasta")) + list(profiles_dir.glob("*.fa"))
    
    with open(output_path, 'w') as f:
        f.write("# Provenance Report\n")
        f.write(f"# Generated from {profiles_dir}\n")
        f.write(f"# Merge history: {merge_history_path}\n\n")
        
        for fasta_path in sorted(fasta_files):
            profile_name = fasta_path.stem
            seq_ids = extract_sequence_ids(fasta_path)
            origins = trace_origins(profile_name, merge_history)
            
            f.write(f"## Profile: {profile_name}\n")
            f.write(f"Sequences ({len(seq_ids)}): {', '.join(seq_ids)}\n")
            f.write(f"Original profiles ({len(origins)}): {', '.join(sorted(origins))}\n")
            
            if profile_name in merge_history:
                f.write("Merge tree:\n")
                f.write(build_merge_tree_text(profile_name, merge_history))
                f.write("\n")
            
            f.write("\n")



# ============================================================
# Command-line interface
# ============================================================

def main():
    """Command-line interface for provenance utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Track sequence origins through merge pipeline"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # cls command
    cls_parser = subparsers.add_parser('cls', help='Generate .cls file')
    cls_parser.add_argument('profiles_dir', type=Path, help='Directory with profiles')
    cls_parser.add_argument('-o', '--output', type=Path, default=Path('profiles.cls'),
                           help='Output .cls file')
    
    # trace command
    trace_parser = subparsers.add_parser('trace', help='Trace profile origins')
    trace_parser.add_argument('profile', help='Profile name to trace')
    trace_parser.add_argument('history', type=Path, help='merge_history.tsv file')
    trace_parser.add_argument('--tree', action='store_true', help='Show merge tree')
    
    # report command
    report_parser = subparsers.add_parser('report', help='Generate full provenance report')
    report_parser.add_argument('profiles_dir', type=Path, help='Directory with profiles')
    report_parser.add_argument('history', type=Path, help='merge_history.tsv file')
    report_parser.add_argument('-o', '--output', type=Path, default=Path('provenance.txt'),
                              help='Output report file')
    
    args = parser.parse_args()
    
    if args.command == 'cls':
        profile_map = generate_cls_file(args.profiles_dir, args.output)
        print(f"Generated {args.output} with {len(profile_map)} profiles")
        
    elif args.command == 'trace':
        merge_history = load_merge_history(args.history)
        origins = trace_origins(args.profile, merge_history)
        
        print(f"Profile: {args.profile}")
        print(f"Original profiles ({len(origins)}): {', '.join(sorted(origins))}")
        
        if args.tree and args.profile in merge_history:
            print("\nMerge tree:")
            print(build_merge_tree_text(args.profile, merge_history))
            
    elif args.command == 'report':
        generate_full_provenance(args.profiles_dir, args.history, args.output)
        print(f"Generated provenance report: {args.output}")
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
