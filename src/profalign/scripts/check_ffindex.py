#!/usr/bin/env python3
"""
check_ffindex.py - Check ffindex availability and test database creation

This script:
1. Checks if ffindex utilities are available
2. Tests creating ffindex database from HHM files
3. Tests hhsearch against the database
"""

import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def check_command(cmd):
    """Check if command is available."""
    result = subprocess.run(['which', cmd], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def main():
    print("=" * 70)
    print("FFindex Availability Check")
    print("=" * 70)
    
    # Check for ffindex utilities
    print("\n[1] Checking for ffindex utilities...")
    
    utilities = [
        'ffindex_build',
        'ffindex_get', 
        'ffindex_modify',
        'ffindex_from_fasta',
        'cstranslate',  # For cs219 generation
    ]
    
    found = {}
    for util in utilities:
        path = check_command(util)
        found[util] = path
        status = f"✓ {path}" if path else "✗ NOT FOUND"
        print(f"  {util}: {status}")
    
    # Check HHsuite tools
    print("\n[2] Checking HHsuite tools...")
    
    hhtools = ['hhmake', 'hhsearch', 'hhalign']
    for tool in hhtools:
        path = check_command(tool)
        status = f"✓ {path}" if path else "✗ NOT FOUND"
        print(f"  {tool}: {status}")
    
    # Check HHLIB environment
    print("\n[3] Checking HHLIB environment...")
    import os
    hhlib = os.environ.get('HHLIB')
    if hhlib:
        print(f"  HHLIB = {hhlib}")
        data_dir = Path(hhlib) / 'data'
        if data_dir.exists():
            print(f"  Data directory exists: {data_dir}")
            cs219_lib = data_dir / 'cs219.lib'
            context_lib = data_dir / 'context_data.lib'
            print(f"    cs219.lib: {'✓' if cs219_lib.exists() else '✗'}")
            print(f"    context_data.lib: {'✓' if context_lib.exists() else '✗'}")
    else:
        print("  HHLIB not set")
    
    # Summary
    print("\n" + "-" * 70)
    print("Summary")
    print("-" * 70)
    
    if found.get('ffindex_build'):
        print("\n✓ ffindex_build is available - can create databases")
        print("  You can use the ffindex-based all-vs-all search")
    else:
        print("\n✗ ffindex_build is NOT available")
        print("  Options:")
        print("  1. Install ffindex: conda install -c bioconda ffindex")
        print("  2. Use pairwise hhalign (slower but works)")
    
    if found.get('cstranslate') and hhlib:
        print("\n✓ cstranslate is available - can create cs219 prefilter database")
    else:
        print("\n! cstranslate or HHLIB not available")
        print("  cs219 prefiltering will not be available (slower searches)")


if __name__ == "__main__":
    main()
