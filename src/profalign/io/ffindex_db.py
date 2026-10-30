"""
ffindex_db.py - FFindex database operations for HHsuite

This module provides functions to:
1. Create ffindex databases from HHM files
2. Generate cs219 prefilter databases (optional, speeds up search)
3. Run hhsearch against ffindex databases
4. Parse results for all-vs-all comparisons

FFindex database structure:
    db_hhm.ffdata   - concatenated HHM files
    db_hhm.ffindex  - index file (name, offset, length)
    db_cs219.ffdata - prefilter database (optional)
    db_cs219.ffindex

Reference: HHsuite 3.x documentation
"""

import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FFindexError(Exception):
    """Exception for ffindex operations."""
    pass


def check_ffindex_tools() -> Dict[str, Optional[str]]:
    """
    Check availability of ffindex tools.
    
    Returns:
        Dictionary mapping tool name to path (or None if not found)
    """
    tools = ['ffindex_build', 'ffindex_get', 'ffindex_modify', 'cstranslate']
    result = {}
    
    for tool in tools:
        path = shutil.which(tool)
        result[tool] = path
    
    return result


def get_hhlib_path() -> Optional[Path]:
    """
    Get HHLIB path from environment or try to detect it.
    
    Returns:
        Path to HHLIB directory or None
    """
    # Check environment
    hhlib = os.environ.get('HHLIB')
    if hhlib:
        hhlib_path = Path(hhlib)
        if (hhlib_path / 'data' / 'cs219.lib').exists():
            return hhlib_path
    
    # Try to detect from conda
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_path = Path(conda_prefix)
        if (conda_path / 'data' / 'cs219.lib').exists():
            return conda_path
    
    return None


def run_command(cmd: List[str], description: str = "") -> subprocess.CompletedProcess:
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command and arguments
        description: Description for error messages
        
    Returns:
        CompletedProcess result
    """
    logger.debug(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        raise FFindexError(f"{description} failed: {error_msg}")
    
    return result


def create_cs219_database(a3m_database_basename: Union[str, Path],
                          output_basename: Union[str, Path] = None,
                          hhlib_path: Union[str, Path] = None) -> Tuple[Path, Path]:
    """
    Create cs219 prefilter database from A3M database.
    
    This is REQUIRED for hhsearch in HHsuite 3.x.
    
    Args:
        a3m_database_basename: Base name of A3M ffindex database
        output_basename: Base name for output (default: same as input)
        hhlib_path: Path to HHLIB (default: from environment)
        
    Returns:
        Tuple of (data_path, index_path)
    """
    a3m_database_basename = Path(a3m_database_basename)
    
    if output_basename is None:
        output_basename = a3m_database_basename
    output_basename = Path(output_basename)
    
    # Get HHLIB
    if hhlib_path is None:
        hhlib_path = get_hhlib_path()
    if hhlib_path is None:
        raise FFindexError("HHLIB not found. Set HHLIB environment variable.")
    hhlib_path = Path(hhlib_path)
    
    cs219_lib = hhlib_path / 'data' / 'cs219.lib'
    context_lib = hhlib_path / 'data' / 'context_data.lib'
    
    if not cs219_lib.exists() or not context_lib.exists():
        raise FFindexError(f"cs219.lib or context_data.lib not found in {hhlib_path}/data/")
    
    # Input files
    a3m_data = Path(str(a3m_database_basename) + "_a3m.ffdata")
    a3m_index = Path(str(a3m_database_basename) + "_a3m.ffindex")
    
    if not a3m_data.exists():
        raise FFindexError(f"A3M data file not found: {a3m_data}")
    if not a3m_index.exists():
        raise FFindexError(f"A3M index file not found: {a3m_index}")
    
    # Debug: check A3M database
    logger.debug(f"A3M database: {a3m_data} ({a3m_data.stat().st_size} bytes)")
    logger.debug(f"A3M index: {a3m_index} ({a3m_index.stat().st_size} bytes)")
    
    # Output files  
    cs219_data = Path(str(output_basename) + "_cs219.ffdata")
    cs219_index = Path(str(output_basename) + "_cs219.ffindex")
    
    # Run cstranslate
    # Note: -f for single file input, --ffindex for database input
    cmd = [
        'cstranslate',
        '-A', str(cs219_lib),
        '-D', str(context_lib),
        '-x', '0.3',
        '-c', '4',
        '--ffindex',
        '-i', str(a3m_database_basename) + "_a3m",
        '-o', str(output_basename) + "_cs219",
        '-I', 'a3m',
        '-b'
    ]
    
    logger.debug(f"cstranslate command: {' '.join(cmd)}")
    
    run_command(cmd, "cstranslate")
    
    logger.info(f"Created cs219 database: {output_basename}_cs219")
    
    return cs219_data, cs219_index


def build_hhsuite_database(msa_files: List[Union[str, Path]],
                           output_basename: Union[str, Path],
                           create_cs219: bool = True,
                           add_consensus: bool = False) -> Dict[str, Path]:
    """
    Build complete HHsuite database from MSA files.
    
    Creates:
    - A3M database (MSAs) - required for cs219
    - HHM database (profiles) - required for hhsearch
    - CS219 database (prefilter) - required for hhsearch in HHsuite 3.x
    
    Args:
        msa_files: List of MSA files (FASTA format)
        output_basename: Base name for all database files
        create_cs219: Create cs219 prefilter database (required for hhsearch!)
        add_consensus: Add consensus (default False - not recommended for database)
        
    Returns:
        Dictionary with paths to created databases
    """
    output_basename = Path(output_basename)
    output_basename.parent.mkdir(parents=True, exist_ok=True)
    
    result = {}
    
    logger.info(f"Building HHsuite database from {len(msa_files)} MSA files...")
    
    from hhsuite import hhmake
    from formats import read_fasta, write_fasta
    from merge import add_consensus_to_alignment
    
    # Build name mapping: first_seq_id -> profile_name
    name_mapping = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        hhm_files = []
        a3m_files = []
        
        # Step 1: Create A3M and HHM files for each MSA
        for msa_file in msa_files:
            msa_file = Path(msa_file)
            msa_name = msa_file.stem
            
            # Read original alignment (don't modify!)
            alignment = read_fasta(msa_file)
            
            # Record mapping from first sequence ID to profile name
            if alignment.sequences:
                first_seq_id = alignment.sequences[0].id
                name_mapping[first_seq_id] = msa_name
            
            # Optionally add consensus (not recommended)
            if add_consensus:
                alignment = add_consensus_to_alignment(alignment, f"{msa_name}@_cons")
            
            # Write A3M
            a3m_file = tmpdir / f"{msa_name}.a3m"
            write_fasta(alignment, a3m_file)
            a3m_files.append(a3m_file)
            
            # Create HHM from the same A3M
            hhm_file = tmpdir / f"{msa_name}.hhm"
            hhmake(a3m_file, hhm_file, name=msa_name)
            hhm_files.append(hhm_file)
            
            logger.debug(f"  Created profile: {msa_name}")
        
        # Step 2: Build A3M database (needed for cs219)
        a3m_data, a3m_index = create_ffindex_from_files(
            a3m_files, 
            str(output_basename) + "_a3m"
        )
        result['a3m_data'] = a3m_data
        result['a3m_index'] = a3m_index
        logger.info(f"  Created A3M database: {len(a3m_files)} entries")
        
        # Step 3: Build HHM database
        hhm_data, hhm_index = create_ffindex_from_files(
            hhm_files,
            str(output_basename) + "_hhm"  
        )
        result['hhm_data'] = hhm_data
        result['hhm_index'] = hhm_index
        logger.info(f"  Created HHM database: {len(hhm_files)} profiles")
    
    # Step 4: Create CS219 (required for hhsearch in HHsuite 3.x!)
    if create_cs219:
        try:
            cs219_data, cs219_index = create_cs219_database(output_basename)
            result['cs219_data'] = cs219_data
            result['cs219_index'] = cs219_index
            logger.info("  Created CS219 prefilter database")
        except FFindexError as e:
            logger.error(f"  Failed to create CS219 database: {e}")
            logger.error("  hhsearch will NOT work without cs219 in HHsuite 3.x")
            raise
    
    result['basename'] = output_basename
    result['num_profiles'] = len(msa_files)
    result['name_mapping'] = name_mapping  # For resolving target_ids in HHR
    
    return result


def create_ffindex_from_files(files: List[Path], 
                               output_basename: str) -> Tuple[Path, Path]:
    """
    Create ffindex database from list of files.
    
    Args:
        files: List of file paths
        output_basename: Output base name (will add .ffdata/.ffindex)
        
    Returns:
        Tuple of (data_path, index_path)
    """
    data_path = Path(output_basename + ".ffdata")
    index_path = Path(output_basename + ".ffindex")
    
    # Create file list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for file in files:
            f.write(str(file) + '\n')
        file_list = f.name
    
    try:
        cmd = ['ffindex_build', '-s', str(data_path), str(index_path), '-f', file_list]
        run_command(cmd, "ffindex_build")
    finally:
        os.unlink(file_list)
    
    return data_path, index_path


def run_hhsearch_against_db(query_hhm: Union[str, Path],
                            database_basename: Union[str, Path],
                            output_hhr: Union[str, Path],
                            num_hits: int = 0,
                            evalue: float = 1e10,
                            use_cs219: bool = True,
                            cpu: int = 1) -> Path:
    """
    Run hhsearch against ffindex database.
    
    Args:
        query_hhm: Query HHM file
        database_basename: Base name of database (without _hhm.ffdata etc)
        output_hhr: Output HHR file
        num_hits: Maximum number of hits (0 = all)
        evalue: E-value threshold
        use_cs219: Use cs219 prefilter if available
        cpu: Number of CPUs
        
    Returns:
        Path to output HHR file
    """
    query_hhm = Path(query_hhm)
    database_basename = Path(database_basename)
    output_hhr = Path(output_hhr)
    
    # Check database files exist and have content
    hhm_data = Path(str(database_basename) + "_hhm.ffdata")
    hhm_index = Path(str(database_basename) + "_hhm.ffindex")
    
    if not hhm_data.exists():
        raise FFindexError(f"Database data file not found: {hhm_data}")
    if not hhm_index.exists():
        raise FFindexError(f"Database index file not found: {hhm_index}")
    
    # Check files are not empty
    if hhm_data.stat().st_size == 0:
        raise FFindexError(f"Database data file is empty: {hhm_data}")
    if hhm_index.stat().st_size == 0:
        raise FFindexError(f"Database index file is empty: {hhm_index}")
    
    logger.debug(f"Database files: {hhm_data} ({hhm_data.stat().st_size} bytes), "
                 f"{hhm_index} ({hhm_index.stat().st_size} bytes)")
    
    # Build command
    cmd = [
        'hhsearch',
        '-i', str(query_hhm),
        '-d', str(database_basename),
        '-o', str(output_hhr),
        '-e', str(evalue),
        '-cpu', str(cpu),
        '-v', '0'
    ]
    
    if num_hits > 0:
        cmd.extend(['-Z', str(num_hits), '-B', str(num_hits)])
    
    # Check if cs219 exists
    cs219_data = Path(str(database_basename) + "_cs219.ffdata")
    if not use_cs219 or not cs219_data.exists():
        cmd.append('-noprefilter')
    
    run_command(cmd, f"hhsearch ({query_hhm.name})")
    
    return output_hhr


def run_all_vs_all_hhsearch(msa_files: List[Union[str, Path]],
                            work_dir: Union[str, Path],
                            use_cs219: bool = True,
                            add_consensus: bool = False,
                            cpu: int = 1) -> Dict:
    """
    Run all-vs-all comparison using hhsearch with ffindex database.
    
    This is much faster than pairwise hhalign for large numbers of profiles.
    
    Note: add_consensus should be False (default) because consensus interferes
    with hhmake's match column detection.
    
    Args:
        msa_files: List of MSA files
        work_dir: Working directory for database and results
        use_cs219: Use cs219 prefilter
        add_consensus: Add consensus to MSAs (default False - not recommended)
        cpu: Number of CPUs for hhsearch
        
    Returns:
        Dictionary with:
        - 'hhr_files': List of HHR file paths
        - 'database': Database basename
        - 'profile_names': List of profile names
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    msa_files = [Path(f) for f in msa_files]
    profile_names = [f.stem for f in msa_files]
    
    logger.info(f"Running all-vs-all hhsearch on {len(msa_files)} profiles")
    
    # Step 1: Build database (no consensus)
    db_basename = work_dir / "profiles_db"
    
    db_info = build_hhsuite_database(
        msa_files,
        db_basename,
        create_cs219=use_cs219,
        add_consensus=add_consensus
    )
    
    # Step 2: Create query HHMs (same as database)
    from hhsuite import hhmake
    from formats import read_fasta, write_fasta
    from merge import add_consensus_to_alignment
    
    query_dir = work_dir / "queries"
    query_dir.mkdir(exist_ok=True)
    
    # Build name mapping for queries too
    name_mapping = db_info.get('name_mapping', {})
    
    hhm_files = []
    for msa_file in msa_files:
        msa_name = msa_file.stem
        
        # Read alignment (don't modify!)
        alignment = read_fasta(msa_file)
        
        # Record mapping
        if alignment.sequences:
            first_seq_id = alignment.sequences[0].id
            name_mapping[first_seq_id] = msa_name
        
        if add_consensus:
            alignment = add_consensus_to_alignment(alignment, f"{msa_name}@_cons")
        
        # Write temp MSA
        temp_msa = query_dir / f"{msa_name}.a3m"
        write_fasta(alignment, temp_msa)
        
        # Create HHM
        hhm_file = query_dir / f"{msa_name}.hhm"
        hhmake(temp_msa, hhm_file, name=msa_name)
        hhm_files.append(hhm_file)
    
    # Step 3: Run hhsearch for each query
    results_dir = work_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    hhr_files = []
    for i, hhm_file in enumerate(hhm_files):
        profile_name = profile_names[i]
        hhr_file = results_dir / f"{profile_name}.hhr"
        
        logger.info(f"  [{i+1}/{len(hhm_files)}] Searching: {profile_name}")
        
        run_hhsearch_against_db(
            hhm_file,
            db_basename,
            hhr_file,
            num_hits=len(msa_files),
            use_cs219=use_cs219,
            cpu=cpu
        )
        
        hhr_files.append(hhr_file)
    
    return {
        'hhr_files': hhr_files,
        'database': db_basename,
        'profile_names': profile_names,
        'num_profiles': len(msa_files),
        'name_mapping': name_mapping  # Map first_seq_id -> profile_name
    }
