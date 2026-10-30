#!/usr/bin/env python3
"""
test_hhsuite.py - Test script to verify HH-suite integration

This script:
1. Reads a FASTA MSA file
2. Creates an HHM profile using hhmake
3. Reports success/failure

Usage:
    python test_hhsuite.py <fasta_file>
    
Example:
    python test_hhsuite.py ../data/msa/C25_Cam1_1.fasta
"""

import sys
import tempfile
from pathlib import Path

# Add lib directory to path

from formats import read_fasta, get_alignment_stats
from hhsuite import check_hhsuite_installation, hhmake, HHSuiteError


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_hhsuite.py <fasta_file>")
        print("Example: python test_hhsuite.py ../data/msa/C25_Cam1_1.fasta")
        sys.exit(1)
    
    fasta_file = Path(sys.argv[1])
    
    if not fasta_file.exists():
        print(f"ERROR: File not found: {fasta_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("HH-suite Integration Test")
    print("=" * 60)
    
    # Step 1: Check HH-suite installation
    print("\n[1] Checking HH-suite installation...")
    tools = check_hhsuite_installation()
    
    all_ok = True
    for tool, info in tools.items():
        status = "OK" if info['available'] else "NOT FOUND"
        print(f"    {tool}: {status}")
        if info['available']:
            print(f"        Path: {info['path']}")
        else:
            all_ok = False
    
    if not all_ok:
        print("\nERROR: Some HH-suite tools are not available!")
        print("Please ensure HH-suite is installed and in your PATH.")
        sys.exit(1)
    
    # Step 2: Read FASTA file
    print(f"\n[2] Reading FASTA file: {fasta_file}")
    try:
        alignment = read_fasta(fasta_file)
        stats = get_alignment_stats(alignment)
        
        print(f"    Sequences: {stats['num_sequences']}")
        print(f"    Alignment length: {stats['alignment_length']}")
        print(f"    Avg sequence length (ungapped): {stats['avg_seq_length']:.1f}")
        
        # Show first few sequence IDs
        ids = alignment.get_ids()[:5]
        print(f"    First sequences: {', '.join(ids)}")
        if len(alignment) > 5:
            print(f"    ... and {len(alignment) - 5} more")
            
    except Exception as e:
        print(f"ERROR reading FASTA: {e}")
        sys.exit(1)
    
    # Step 3: Create HHM profile
    print(f"\n[3] Creating HHM profile with hhmake...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_hhm = tmpdir / f"{fasta_file.stem}.hhm"
        
        try:
            hhmake(fasta_file, output_hhm)
            
            # Check output file
            if output_hhm.exists():
                size = output_hhm.stat().st_size
                print(f"    Output: {output_hhm.name}")
                print(f"    Size: {size} bytes")
                
                # Read first few lines to verify
                with open(output_hhm, 'r') as f:
                    lines = f.readlines()[:10]
                    print("    First lines of HHM file:")
                    for line in lines:
                        print(f"      {line.rstrip()}")
            else:
                print("    ERROR: Output file was not created!")
                sys.exit(1)
                
        except HHSuiteError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    # Success
    print("\n" + "=" * 60)
    print("SUCCESS: All tests passed!")
    print("=" * 60)
    print("\nYour environment is ready for prof_align.")


if __name__ == "__main__":
    main()
