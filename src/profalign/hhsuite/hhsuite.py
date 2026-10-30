"""
hhsuite.py - Wrapper functions for HH-suite tools

Provides Python interfaces for:
- hhmake: Build HMM profile from MSA
- hhsearch: Search profile against profile database
- hhalign: Align two profiles

Requires HH-suite 3.x installed and in PATH.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class HHSuiteError(Exception):
    """Exception raised when HH-suite tool fails."""
    pass


def check_hhsuite_installation() -> dict:
    """
    Check if HH-suite tools are available.
    
    Returns:
        Dictionary with tool paths and availability status
    """
    tools = ['hhmake', 'hhsearch', 'hhalign']
    result = {}
    
    for tool in tools:
        path = shutil.which(tool)
        result[tool] = {
            'available': path is not None,
            'path': path
        }
    
    return result


def run_command(cmd: List[str], description: str = "") -> subprocess.CompletedProcess:
    """
    Run external command with error handling.
    
    Args:
        cmd: Command and arguments as list
        description: Human-readable description for logging
        
    Returns:
        CompletedProcess object
        
    Raises:
        HHSuiteError: If command fails
    """
    logger.debug(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise HHSuiteError(
                f"{description or cmd[0]} failed (exit code {result.returncode}): {error_msg}"
            )
        
        return result
        
    except FileNotFoundError:
        raise HHSuiteError(f"Tool not found: {cmd[0]}. Is HH-suite installed?")


def hhmake(input_msa: Union[str, Path],
           output_hhm: Union[str, Path],
           name: Optional[str] = None,
           add_cons: bool = False,
           seq_id: int = 99,
           coverage: float = 0.0,
           m_option: str = "first") -> Path:
    """
    Build HMM profile from MSA using hhmake.
    
    Args:
        input_msa: Input MSA file (FASTA/A3M format)
        output_hhm: Output HHM profile file
        name: Profile name (default: derived from input filename)
        add_cons: Add consensus sequence to profile
        seq_id: Maximum sequence identity for filtering (default: 99, matches Yuri's -id 99)
        coverage: Minimum coverage threshold (default: 0.0)
        m_option: Match state definition - 'first' for first sequence (default, matches Yuri),
                  or percentage like '50' for 50% occupancy threshold
        
    Returns:
        Path to output HHM file
        
    Raises:
        HHSuiteError: If hhmake fails
    """
    input_msa = Path(input_msa)
    output_hhm = Path(output_hhm)
    
    if name is None:
        name = input_msa.stem
    
    cmd = [
        'hhmake',
        '-i', str(input_msa),
        '-o', str(output_hhm),
        '-name', name,
        '-id', str(seq_id),
        '-M', m_option,
        '-v', '0'  # Suppress verbose output
    ]
    
    if add_cons:
        cmd.append('-add_cons')
    
    if coverage > 0:
        cmd.extend(['-cov', str(coverage)])
    
    run_command(cmd, f"hhmake ({input_msa.name})")
    
    return output_hhm


def hhsearch(query_hhm: Union[str, Path],
             database: Union[str, Path],
             output_hhr: Union[str, Path],
             num_hits: int = 100,
             evalue: float = 1e10,
             global_alignment: bool = False,
             realign: bool = False,
             seq_id: int = 99,
             m_occupancy: float = 0.4999,
             cpu: int = 1) -> Path:
    """
    Search profile against profile database using hhsearch.
    
    Args:
        query_hhm: Query HHM profile file
        database: Database HHM file (concatenated profiles)
        output_hhr: Output results file (.hhr)
        num_hits: Maximum number of hits to report (default: 100)
        evalue: E-value cutoff (default: 1e10, effectively no cutoff)
        global_alignment: Use global alignment mode (default: local)
        realign: Realign hits with MAC algorithm (default: False)
        seq_id: Maximum sequence identity for prefilter (default: 99, matches Yuri's -id)
        m_occupancy: Match column occupancy threshold (default: 0.4999, matches Yuri's -M)
        cpu: Number of CPUs (default: 1)
        
    Returns:
        Path to output HHR file
        
    Raises:
        HHSuiteError: If hhsearch fails
    """
    query_hhm = Path(query_hhm)
    database = Path(database)
    output_hhr = Path(output_hhr)
    
    cmd = [
        'hhsearch',
        '-i', str(query_hhm),
        '-d', str(database),
        '-o', str(output_hhr),
        '-z', str(num_hits),     # Number of hits to show
        '-Z', str(num_hits),     # Number of hits to report
        '-b', str(num_hits),     # Number of alignments to show
        '-B', str(num_hits),     # Number of alignments to report
        '-e', str(evalue),
        '-id', str(seq_id),      # Sequence identity for prefilter (Yuri's -id 99)
        '-M', str(m_occupancy),  # Match column occupancy (Yuri's -M 0.4999)
        '-cpu', str(cpu),
        '-v', '0',
        '-nocons'                # Don't add consensus (matches Yuri's -nocons)
    ]
    
    if global_alignment:
        cmd.append('-glob')
    else:
        cmd.append('-loc')
    
    if not realign:
        cmd.append('-norealign')
    
    run_command(cmd, f"hhsearch ({query_hhm.name})")
    
    return output_hhr


def hhalign(query: Union[str, Path],
            template: Union[str, Path],
            output: Union[str, Path],
            global_alignment: bool = True,
            output_format: str = "hhr",
            m_option: str = "first",
            seq_id: int = 100,
            diff: str = "inf") -> Path:
    """
    Align two profiles/MSAs using hhalign.
    
    Args:
        query: Query profile (HHM) or MSA (FASTA/A3M)
        template: Template profile (HHM) or MSA (FASTA/A3M)
        output: Output file path
        global_alignment: Use global alignment (default: True)
        output_format: Output format - 'hhr', 'a3m', or 'a2m' (default: 'hhr')
        m_option: Match column definition - 'first', 'a2m', or percentage (default: 'first', matches Yuri)
        seq_id: Maximum sequence identity (default: 100, matches Yuri's -id 100)
        diff: Diversity parameter (default: 'inf', matches Yuri's -diff inf)
        
    Returns:
        Path to output file
        
    Raises:
        HHSuiteError: If hhalign fails
    """
    query = Path(query)
    template = Path(template)
    output = Path(output)
    
    cmd = [
        'hhalign',
        '-i', str(query),
        '-t', str(template),
        '-o', str(output),
        '-M', m_option,      # Use first sequence to define match columns (Yuri's -M first)
        '-id', str(seq_id),  # Maximum sequence identity (Yuri's -id 100)
        '-diff', diff,       # Diversity parameter (Yuri's -diff inf)
        '-v', '0'
    ]
    
    if global_alignment:
        cmd.append('-glob')
    else:
        cmd.append('-loc')
    
    run_command(cmd, f"hhalign ({query.name} vs {template.name})")
    
    return output


