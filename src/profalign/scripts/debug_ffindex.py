#!/usr/bin/env python3
"""
debug_ffindex.py - Debug ffindex database creation step by step
"""

import sys
import subprocess
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'lib'))

from formats import read_fasta, write_fasta
from hhsuite import hhmake

def run_cmd(cmd, desc):
    """Run command and print output."""
    print(f"\n>>> {desc}")
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"    Return code: {result.returncode}")
    if result.stdout:
        print(f"    STDOUT: {result.stdout[:500]}")
    if result.stderr:
        print(f"    STDERR: {result.stderr[:500]}")
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_ffindex.py <msa_dir>")
        sys.exit(1)
    
    msa_dir = Path(sys.argv[1])
    msa_files = sorted(msa_dir.glob("*.fasta"))[:3]  # Just 3 files for debug
    
    print("=" * 70)
    print("FFindex Debug")
    print("=" * 70)
    
    print(f"\nUsing {len(msa_files)} MSA files")
    for f in msa_files:
        print(f"  {f}")
    
    # Check HHLIB
    hhlib = os.environ.get('HHLIB')
    print(f"\nHHLIB = {hhlib}")
    if hhlib:
        cs219_lib = Path(hhlib) / 'data' / 'cs219.lib'
        context_lib = Path(hhlib) / 'data' / 'context_data.lib'
        print(f"  cs219.lib exists: {cs219_lib.exists()}")
        print(f"  context_data.lib exists: {context_lib.exists()}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"\nWorking directory: {tmpdir}")
        
        # Step 1: Create A3M files (original, no modifications)
        print("\n" + "-" * 70)
        print("Step 1: Create A3M files")
        print("-" * 70)
        
        a3m_files = []
        for msa_file in msa_files:
            name = msa_file.stem
            
            # Read original alignment - no modifications!
            ali = read_fasta(msa_file)
            
            # Write A3M
            a3m_file = tmpdir / f"{name}.a3m"
            write_fasta(ali, a3m_file)
            a3m_files.append(a3m_file)
            print(f"  Created {a3m_file.name} ({a3m_file.stat().st_size} bytes)")
        
        # Step 2: Create A3M ffindex database
        print("\n" + "-" * 70)
        print("Step 2: Create A3M ffindex database")
        print("-" * 70)
        
        file_list = tmpdir / "a3m_files.txt"
        with open(file_list, 'w') as f:
            for a3m in a3m_files:
                f.write(str(a3m) + '\n')
        
        a3m_data = tmpdir / "test_a3m.ffdata"
        a3m_index = tmpdir / "test_a3m.ffindex"
        
        result = run_cmd([
            'ffindex_build', '-s',
            str(a3m_data), str(a3m_index),
            '-f', str(file_list)
        ], "ffindex_build (a3m)")
        
        if a3m_data.exists():
            print(f"  A3M data: {a3m_data} ({a3m_data.stat().st_size} bytes)")
        if a3m_index.exists():
            print(f"  A3M index: {a3m_index} ({a3m_index.stat().st_size} bytes)")
            print(f"  Index contents:")
            print(open(a3m_index).read())
        
        # Step 3: Run cstranslate
        print("\n" + "-" * 70)
        print("Step 3: Run cstranslate")
        print("-" * 70)
        
        if hhlib:
            cs219_lib = Path(hhlib) / 'data' / 'cs219.lib'
            context_lib = Path(hhlib) / 'data' / 'context_data.lib'
            
            db_basename = tmpdir / "test"  # cstranslate expects _a3m suffix
            
            result = run_cmd([
                'cstranslate',
                '-A', str(cs219_lib),
                '-D', str(context_lib),
                '-x', '0.3',
                '-c', '4',
                '--ffindex',
                '-i', str(db_basename) + "_a3m",
                '-o', str(db_basename) + "_cs219",
                '-I', 'a3m',
                '-b'
            ], "cstranslate")
            
            cs219_data = tmpdir / "test_cs219.ffdata"
            cs219_index = tmpdir / "test_cs219.ffindex"
            
            if cs219_data.exists():
                print(f"  CS219 data: {cs219_data} ({cs219_data.stat().st_size} bytes)")
            else:
                print(f"  ERROR: CS219 data not created!")
                
            if cs219_index.exists():
                print(f"  CS219 index: {cs219_index} ({cs219_index.stat().st_size} bytes)")
            else:
                print(f"  ERROR: CS219 index not created!")
        else:
            print("  HHLIB not set, skipping cstranslate")
        
        # Step 4: Create HHM files
        print("\n" + "-" * 70)
        print("Step 4: Create HHM files")
        print("-" * 70)
        
        hhm_files = []
        for a3m_file in a3m_files:
            name = a3m_file.stem
            hhm_file = tmpdir / f"{name}.hhm"
            hhmake(a3m_file, hhm_file, name=name)
            hhm_files.append(hhm_file)
            print(f"  Created {hhm_file.name} ({hhm_file.stat().st_size} bytes)")
        
        # Step 5: Create HHM ffindex database
        print("\n" + "-" * 70)
        print("Step 5: Create HHM ffindex database")
        print("-" * 70)
        
        file_list = tmpdir / "hhm_files.txt"
        with open(file_list, 'w') as f:
            for hhm in hhm_files:
                f.write(str(hhm) + '\n')
        
        hhm_data = tmpdir / "test_hhm.ffdata"
        hhm_index = tmpdir / "test_hhm.ffindex"
        
        result = run_cmd([
            'ffindex_build', '-s',
            str(hhm_data), str(hhm_index),
            '-f', str(file_list)
        ], "ffindex_build (hhm)")
        
        if hhm_data.exists():
            print(f"  HHM data: {hhm_data} ({hhm_data.stat().st_size} bytes)")
        if hhm_index.exists():
            print(f"  HHM index: {hhm_index} ({hhm_index.stat().st_size} bytes)")
            print(f"  Index contents:")
            print(open(hhm_index).read())
        
        # Step 6: Run hhsearch
        print("\n" + "-" * 70)
        print("Step 6: Run hhsearch")
        print("-" * 70)
        
        query_hhm = hhm_files[0]
        output_hhr = tmpdir / "test.hhr"
        db_basename = tmpdir / "test"
        
        print(f"  Query: {query_hhm}")
        print(f"  Database: {db_basename}")
        print(f"  Files needed:")
        print(f"    {db_basename}_hhm.ffdata: {(tmpdir / 'test_hhm.ffdata').exists()}")
        print(f"    {db_basename}_hhm.ffindex: {(tmpdir / 'test_hhm.ffindex').exists()}")
        print(f"    {db_basename}_cs219.ffdata: {(tmpdir / 'test_cs219.ffdata').exists()}")
        print(f"    {db_basename}_cs219.ffindex: {(tmpdir / 'test_cs219.ffindex').exists()}")
        
        result = run_cmd([
            'hhsearch',
            '-i', str(query_hhm),
            '-d', str(db_basename),
            '-o', str(output_hhr),
            '-v', '2'
        ], "hhsearch")
        
        if output_hhr.exists():
            print(f"  Output: {output_hhr} ({output_hhr.stat().st_size} bytes)")
            print(f"  First 500 chars:")
            print(open(output_hhr).read()[:500])
        else:
            print(f"  ERROR: {output_hhr} not created!")


if __name__ == "__main__":
    main()
