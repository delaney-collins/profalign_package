#!/usr/bin/env python3
"""
precluster.py - Pre-clustering of sequences before profile alignment

Clusters raw sequences using MMseqs2 and creates MSA for each cluster
using MUSCLE. Output MSA files can be used as input to prof_align.py.

Workflow:
    Raw FASTA → MMseqs2 clustering → MUSCLE alignment → MSA files

Usage:
    python precluster.py input.fasta output_dir/ [options]

Requirements:
    - mmseqs2 (conda install -c bioconda mmseqs2)
    - muscle v5 (conda install -c bioconda muscle)

Author: profalign team
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_version():
    """Read version from VERSION file."""
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "unknown"


@dataclass
class ClusterConfig:
    """Configuration for pre-clustering."""
    input_fasta: Path
    output_dir: Path
    identity: float = 0.5          # Sequence identity threshold (mmseqs --min-seq-id)
    coverage: float = 0.333        # Coverage threshold (mmseqs -c)
    cov_mode: int = 1              # Coverage mode: 0=longer, 1=target, 2=query
    cluster_mode: int = 2          # Cluster mode: 0=set cover, 1=connected, 2=greedy
    aligner: str = "muscle"        # MSA tool: muscle or mafft
    threads: int = 4               # Number of threads
    min_cluster_size: int = 1      # Minimum sequences per cluster to output
    keep_singletons: bool = True   # Output singleton sequences as single-seq MSAs


def check_dependencies() -> Dict[str, bool]:
    """Check if required tools are available."""
    tools = {
        'mmseqs': False,
        'muscle': False,
        'mafft': False,  # optional
    }
    
    # Check mmseqs (doesn't support --version, just run without args)
    try:
        result = subprocess.run(['mmseqs'], capture_output=True, text=True)
        # mmseqs without args prints help and returns non-zero, but that's OK
        tools['mmseqs'] = 'MMseqs2' in result.stdout or 'MMseqs2' in result.stderr
    except FileNotFoundError:
        tools['mmseqs'] = False
    
    # Check muscle
    try:
        result = subprocess.run(['muscle', '--version'], capture_output=True, text=True)
        tools['muscle'] = result.returncode == 0 or 'muscle' in result.stdout.lower()
    except FileNotFoundError:
        tools['muscle'] = False
    
    # Check mafft
    try:
        result = subprocess.run(['mafft', '--version'], capture_output=True, text=True)
        tools['mafft'] = 'MAFFT' in result.stderr or 'MAFFT' in result.stdout
    except FileNotFoundError:
        tools['mafft'] = False
    
    return tools


def read_fasta(filepath: Path) -> Dict[str, str]:
    """
    Read FASTA file into dictionary.
    
    Returns:
        Dict mapping sequence_id -> sequence
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                # Take first word as ID
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def write_fasta(sequences: Dict[str, str], filepath: Path, wrap: int = 80):
    """Write sequences to FASTA file."""
    with open(filepath, 'w') as f:
        for seq_id, seq in sequences.items():
            f.write(f'>{seq_id}\n')
            # Wrap sequence
            for i in range(0, len(seq), wrap):
                f.write(seq[i:i+wrap] + '\n')


def run_mmseqs_cluster(
    input_fasta: Path,
    output_dir: Path,
    identity: float,
    coverage: float,
    cov_mode: int,
    cluster_mode: int,
    threads: int
) -> Path:
    """
    Run MMseqs2 clustering.
    
    Returns:
        Path to clusters TSV file
    """
    work_dir = output_dir / 'mmseqs_work'
    work_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = work_dir / 'input_db'
    cluster_db = work_dir / 'cluster_db'
    tmp_dir = work_dir / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    
    clusters_tsv = output_dir / 'clusters.tsv'
    
    # Step 1: Create database
    logger.info("Creating MMseqs2 database...")
    cmd = ['mmseqs', 'createdb', str(input_fasta), str(db_path)]
    logger.debug(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs createdb failed: {result.stderr}")
    
    # Step 2: Cluster
    logger.info(f"Clustering (identity={identity}, coverage={coverage})...")
    cmd = [
        'mmseqs', 'cluster',
        str(db_path), str(cluster_db), str(tmp_dir),
        '--min-seq-id', str(identity),
        '-c', str(coverage),
        '--cov-mode', str(cov_mode),
        '--cluster-mode', str(cluster_mode),
        '--threads', str(threads)
    ]
    logger.debug(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs cluster failed: {result.stderr}")
    
    # Step 3: Create TSV output
    logger.info("Generating cluster table...")
    cmd = [
        'mmseqs', 'createtsv',
        str(db_path), str(db_path), str(cluster_db),
        str(clusters_tsv)
    ]
    logger.debug(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs createtsv failed: {result.stderr}")
    
    return clusters_tsv


def parse_clusters(clusters_tsv: Path) -> Dict[str, List[str]]:
    """
    Parse MMseqs2 clusters TSV file.
    
    Format: representative_id<TAB>member_id
    
    Returns:
        Dict mapping representative_id -> list of member_ids
    """
    clusters = {}
    
    with open(clusters_tsv) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep_id, member_id = parts[0], parts[1]
                if rep_id not in clusters:
                    clusters[rep_id] = []
                clusters[rep_id].append(member_id)
    
    return clusters


def run_muscle_align(
    sequences: Dict[str, str],
    output_path: Path,
    threads: int = 1
) -> bool:
    """
    Run MUSCLE alignment on sequences.
    
    Uses different strategies based on number of sequences:
    - 1 sequence: just write as-is
    - 2-5 sequences: muscle -align (accurate)
    - >5 sequences: muscle -super5 (fast)
    
    Returns:
        True if successful
    """
    if len(sequences) == 0:
        return False
    
    if len(sequences) == 1:
        # Singleton - just write as-is
        write_fasta(sequences, output_path)
        return True
    
    # Write input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
        tmp_input = Path(tmp.name)
        for seq_id, seq in sequences.items():
            tmp.write(f'>{seq_id}\n{seq}\n')
    
    try:
        if len(sequences) <= 5:
            # Accurate alignment for small clusters
            cmd = ['muscle', '-align', str(tmp_input), '-output', str(output_path)]
        else:
            # Fast alignment for larger clusters
            cmd = ['muscle', '-super5', str(tmp_input), '-output', str(output_path)]
        
        logger.debug(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"MUSCLE failed: {result.stderr}")
            return False
        
        return True
    
    finally:
        tmp_input.unlink(missing_ok=True)


def run_mafft_align(
    sequences: Dict[str, str],
    output_path: Path,
    threads: int = 1
) -> bool:
    """
    Run MAFFT alignment on sequences.
    
    Returns:
        True if successful
    """
    if len(sequences) == 0:
        return False
    
    if len(sequences) == 1:
        write_fasta(sequences, output_path)
        return True
    
    # Write input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
        tmp_input = Path(tmp.name)
        for seq_id, seq in sequences.items():
            tmp.write(f'>{seq_id}\n{seq}\n')
    
    try:
        cmd = [
            'mafft',
            '--auto',
            '--thread', str(threads),
            '--quiet',
            str(tmp_input)
        ]
        
        logger.debug(f"  Command: {' '.join(cmd)}")
        
        with open(output_path, 'w') as out_f:
            result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            logger.warning(f"MAFFT failed: {result.stderr}")
            return False
        
        return True
    
    finally:
        tmp_input.unlink(missing_ok=True)


def run_precluster(config: ClusterConfig) -> Tuple[List[Path], dict]:
    """
    Main pre-clustering workflow.
    
    Returns:
        Tuple of (list of MSA file paths, statistics dict)
    """
    stats = {
        'input_sequences': 0,
        'num_clusters': 0,
        'singletons': 0,
        'largest_cluster': 0,
        'output_files': 0,
    }
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input sequences
    logger.info(f"Reading input: {config.input_fasta}")
    all_sequences = read_fasta(config.input_fasta)
    stats['input_sequences'] = len(all_sequences)
    logger.info(f"  Found {len(all_sequences)} sequences")
    
    if len(all_sequences) == 0:
        raise ValueError("No sequences found in input file")
    
    # Run MMseqs2 clustering
    clusters_tsv = run_mmseqs_cluster(
        input_fasta=config.input_fasta,
        output_dir=config.output_dir,
        identity=config.identity,
        coverage=config.coverage,
        cov_mode=config.cov_mode,
        cluster_mode=config.cluster_mode,
        threads=config.threads
    )
    
    # Parse clusters
    clusters = parse_clusters(clusters_tsv)
    stats['num_clusters'] = len(clusters)
    logger.info(f"  Found {len(clusters)} clusters")
    
    # Create MSA directory
    msa_dir = config.output_dir / 'msa'
    msa_dir.mkdir(exist_ok=True)
    
    # Process each cluster
    output_files = []
    singletons = 0
    largest = 0
    
    # Choose alignment function
    align_func = run_mafft_align if config.aligner == 'mafft' else run_muscle_align
    
    logger.info(f"Aligning clusters with {config.aligner}...")
    
    for idx, (rep_id, member_ids) in enumerate(sorted(clusters.items(), 
                                                       key=lambda x: -len(x[1])), 1):
        cluster_size = len(member_ids)
        largest = max(largest, cluster_size)
        
        if cluster_size == 1:
            singletons += 1
        
        # Skip if below minimum size (unless keeping singletons)
        if cluster_size < config.min_cluster_size:
            if not (config.keep_singletons and cluster_size == 1):
                continue
        
        # Extract sequences for this cluster
        cluster_seqs = {seq_id: all_sequences[seq_id] 
                       for seq_id in member_ids 
                       if seq_id in all_sequences}
        
        if not cluster_seqs:
            logger.warning(f"  Cluster {idx}: no sequences found")
            continue
        
        # Generate output filename
        # Use representative ID (sanitized) or cluster number
        safe_name = rep_id.replace('/', '_').replace('\\', '_')[:50]
        output_path = msa_dir / f'cluster_{idx:04d}_{safe_name}.fasta'
        
        # Run alignment
        if cluster_size > 1:
            logger.info(f"  Cluster {idx}: {cluster_size} sequences")
        
        success = align_func(cluster_seqs, output_path, config.threads)
        
        if success:
            output_files.append(output_path)
        else:
            logger.warning(f"  Cluster {idx}: alignment failed, writing unaligned")
            write_fasta(cluster_seqs, output_path)
            output_files.append(output_path)
    
    stats['singletons'] = singletons
    stats['largest_cluster'] = largest
    stats['output_files'] = len(output_files)
    
    # Write summary
    summary_path = config.output_dir / 'precluster_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Pre-clustering Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input file: {config.input_fasta}\n")
        f.write(f"Input sequences: {stats['input_sequences']}\n")
        f.write(f"Identity threshold: {config.identity}\n")
        f.write(f"Coverage threshold: {config.coverage}\n")
        f.write(f"Aligner: {config.aligner}\n\n")
        f.write(f"Clusters found: {stats['num_clusters']}\n")
        f.write(f"Singletons: {stats['singletons']}\n")
        f.write(f"Largest cluster: {stats['largest_cluster']}\n")
        f.write(f"Output MSA files: {stats['output_files']}\n\n")
        f.write(f"Output directory: {msa_dir}\n")
    
    return output_files, stats


def main():
    parser = argparse.ArgumentParser(
        description='Pre-cluster sequences before profile alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python precluster.py sequences.fasta output_dir/
  
  # Custom identity threshold
  python precluster.py sequences.fasta output_dir/ --identity 0.7
  
  # Use MAFFT instead of MUSCLE
  python precluster.py sequences.fasta output_dir/ --aligner mafft
  
  # Then run profile alignment
  python prof_align.py output_dir/msa/ results/

Requirements:
  - mmseqs2: conda install -c bioconda mmseqs2
  - muscle:  conda install -c bioconda muscle
  - mafft:   conda install -c bioconda mafft (optional)
"""
    )
    
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {get_version()}')
    parser.add_argument('input_fasta', type=Path,
                        help='Input FASTA file with sequences')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory')
    parser.add_argument('--identity', type=float, default=0.5,
                        help='Sequence identity threshold (default: 0.5)')
    parser.add_argument('--coverage', type=float, default=0.333,
                        help='Coverage threshold (default: 0.333)')
    parser.add_argument('--cov-mode', type=int, default=1, choices=[0, 1, 2],
                        help='Coverage mode: 0=longer, 1=target, 2=query (default: 1)')
    parser.add_argument('--cluster-mode', type=int, default=2, choices=[0, 1, 2],
                        help='Cluster mode: 0=set-cover, 1=connected, 2=greedy (default: 2)')
    parser.add_argument('--aligner', choices=['muscle', 'mafft'], default='muscle',
                        help='MSA tool (default: muscle)')
    parser.add_argument('--threads', type=int, default=4,
                        help='Number of threads (default: 4)')
    parser.add_argument('--min-cluster-size', type=int, default=1,
                        help='Minimum cluster size to output (default: 1)')
    parser.add_argument('--no-singletons', action='store_true',
                        help='Skip singleton clusters')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    logger.info("Checking dependencies...")
    deps = check_dependencies()
    
    if not deps['mmseqs']:
        logger.error("mmseqs2 not found. Install: conda install -c bioconda mmseqs2")
        sys.exit(1)
    
    if args.aligner == 'muscle' and not deps['muscle']:
        logger.error("muscle not found. Install: conda install -c bioconda muscle")
        sys.exit(1)
    
    if args.aligner == 'mafft' and not deps['mafft']:
        logger.error("mafft not found. Install: conda install -c bioconda mafft")
        sys.exit(1)
    
    logger.info("  All dependencies OK")
    
    # Check input file
    if not args.input_fasta.exists():
        logger.error(f"Input file not found: {args.input_fasta}")
        sys.exit(1)
    
    # Create config
    config = ClusterConfig(
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        identity=args.identity,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        cluster_mode=args.cluster_mode,
        aligner=args.aligner,
        threads=args.threads,
        min_cluster_size=args.min_cluster_size,
        keep_singletons=not args.no_singletons
    )
    
    # Run
    try:
        output_files, stats = run_precluster(config)
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("Pre-clustering complete!")
        logger.info(f"  Input sequences: {stats['input_sequences']}")
        logger.info(f"  Clusters: {stats['num_clusters']}")
        logger.info(f"  Singletons: {stats['singletons']}")
        logger.info(f"  Largest cluster: {stats['largest_cluster']}")
        logger.info(f"  Output MSA files: {stats['output_files']}")
        logger.info("")
        logger.info(f"MSA files written to: {config.output_dir / 'msa'}")
        logger.info("")
        logger.info("Next step:")
        logger.info(f"  python prof_align.py {config.output_dir / 'msa'} <results_dir>")
        
    except Exception as e:
        logger.error(f"Pre-clustering failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
