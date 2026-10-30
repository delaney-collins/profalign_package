#!/usr/bin/env python3
"""
Google Cloud Batch executor for profalign pipeline.
Runs hhalign tasks in parallel on Google Cloud.

Profiles are prepared (filtered + consensus) by init, not by executor.
hhalign runs directly on prepared files with parameters: -M first -id 100 -diff inf -glob

Architecture:
  LOCAL                           GCP
  prof_align_cluster.py init  --> upload profiles/ to GCS
  prof_align_cluster.py gen-queue
                                  Cloud Batch: hhalign on prepared files
  prof_align_cluster.py process-results (merges locally)
                              --> upload merged/ to GCS
  ... repeat until converged ...
  prof_align_cluster.py finalize
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from google.cloud import batch_v1
from google.cloud import storage

# ============== CONFIGURATION ==============
PROJECT_ID = "profaligncluster"
REGION = "us-central1"
BUCKET_NAME = "profalign-data-12345"
DOCKER_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/profalign-repo/hhalign:latest"
MAX_PARALLEL_TASKS = 8  # Free tier limit
# ===========================================


def run_command(cmd: list, cwd: str = None) -> tuple:
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.returncode, result.stdout, result.stderr


def upload_directory_to_gcs(local_dir: Path, bucket_name: str, gcs_prefix: str):
    """Upload a local directory to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    count = 0
    for local_file in local_dir.rglob("*"):
        if local_file.is_file():
            relative_path = local_file.relative_to(local_dir)
            blob_name = f"{gcs_prefix}/{relative_path}".replace("\\", "/")
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_file))
            count += 1
    
    print(f"  Uploaded {count} files to gs://{bucket_name}/{gcs_prefix}/")


def download_directory_from_gcs(bucket_name: str, gcs_prefix: str, local_dir: Path):
    """Download a GCS prefix to local directory."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    count = 0
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_path = blob.name[len(gcs_prefix):].lstrip("/")
        local_file = local_dir / relative_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file))
        count += 1
    
    print(f"  Downloaded {count} files from gs://{bucket_name}/{gcs_prefix}/")


def download_file_from_gcs(bucket_name: str, gcs_path: str, local_path: Path):
    """Download a single file from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))


def upload_file_to_gcs(local_path: Path, bucket_name: str, gcs_path: str):
    """Upload a single file to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))


def create_batch_job(job_id: str, tasks: list) -> batch_v1.Job:
    """Create a Cloud Batch job for hhalign tasks."""
    client = batch_v1.BatchServiceClient()
    
    # Script that each task will run
    # NOTE: Input files are already prepared (filtered + consensus) by
    # prof_align_cluster.py init. We just run hhalign directly.
    #
    # hhalign parameters:
    #   -M first    : use first sequence (consensus) to define match states
    #   -id 100     : maximum sequence identity (keep all)
    #   -diff inf   : no sequence weighting
    #   -glob       : global alignment
    #   -v 0        : quiet mode
    task_script = f"""#!/bin/bash
set -e

TASK_INDEX=$BATCH_TASK_INDEX
BUCKET="{BUCKET_NAME}"

# Download task info
gsutil cp gs://$BUCKET/work/batch_tasks/task_$TASK_INDEX.json /tmp/task.json

QUERY=$(cat /tmp/task.json | python3 -c "import sys,json; print(json.load(sys.stdin)['query'])")
TARGET=$(cat /tmp/task.json | python3 -c "import sys,json; print(json.load(sys.stdin)['target'])")
QUERY_PATH=$(cat /tmp/task.json | python3 -c "import sys,json; print(json.load(sys.stdin)['query_path'])")
TARGET_PATH=$(cat /tmp/task.json | python3 -c "import sys,json; print(json.load(sys.stdin)['target_path'])")

# Download prepared input files (already have filter + consensus)
gsutil cp gs://$BUCKET/$QUERY_PATH /tmp/query.fasta
gsutil cp gs://$BUCKET/$TARGET_PATH /tmp/target.fasta

# Run hhalign directly on prepared files
hhalign -i /tmp/query.fasta -t /tmp/target.fasta -o /tmp/result.hhr -M first -id 100 -diff inf -glob -v 0

# Upload HHR result
gsutil cp /tmp/result.hhr gs://$BUCKET/work/hhr/${{QUERY}}_vs_${{TARGET}}.hhr

echo "Task $TASK_INDEX completed: $QUERY vs $TARGET"
"""
    
    # Define the task spec
    task_spec = batch_v1.TaskSpec()
    
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = DOCKER_IMAGE
    runnable.container.entrypoint = "/bin/bash"
    runnable.container.commands = ["-c", task_script]
    
    task_spec.runnables = [runnable]
    task_spec.max_retry_count = 2
    task_spec.max_run_duration = "600s"
    
    # Compute resources
    resources = batch_v1.ComputeResource()
    resources.cpu_milli = 1000  # 1 vCPU
    resources.memory_mib = 2048  # 2 GB
    task_spec.compute_resource = resources
    
    # Task group
    task_group = batch_v1.TaskGroup()
    task_group.task_spec = task_spec
    task_group.task_count = len(tasks)
    task_group.parallelism = min(len(tasks), MAX_PARALLEL_TASKS)
    
    # Allocation policy
    allocation = batch_v1.AllocationPolicy()
    
    instance = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instance.policy = batch_v1.AllocationPolicy.InstancePolicy()
    instance.policy.machine_type = "e2-medium"
    allocation.instances = [instance]
    
    # Logs policy  
    logs = batch_v1.LogsPolicy()
    logs.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING
    
    # Create job
    job = batch_v1.Job()
    job.task_groups = [task_group]
    job.allocation_policy = allocation
    job.logs_policy = logs
    
    # Submit job
    request = batch_v1.CreateJobRequest()
    request.parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    request.job_id = job_id
    request.job = job
    
    return client.create_job(request)


def wait_for_job(job_name: str) -> bool:
    """Wait for a batch job to complete. Returns True if successful."""
    client = batch_v1.BatchServiceClient()
    
    print(f"  Waiting for job to complete...")
    start_time = time.time()
    
    while True:
        job = client.get_job(name=job_name)
        state = job.status.state.name
        elapsed = int(time.time() - start_time)
        
        print(f"    State: {state} ({elapsed}s elapsed)", end="\r")
        
        if state == "SUCCEEDED":
            print(f"    State: {state} - Job completed successfully! ({elapsed}s)")
            return True
        elif state in ["FAILED", "DELETION_IN_PROGRESS"]:
            print(f"    State: {state} - Job failed! ({elapsed}s)")
            return False
        
        time.sleep(10)


def parse_hhr_content(content: str) -> tuple:
    """Parse HHR file content and extract score, probability, aligned_cols, coverage.
    
    Delegates to lib/hhr_parser.py for actual parsing.
    """
    from hhr_parser import parse_hhr_string
    result = parse_hhr_string(content)
    if result.hits:
        hit = result.hits[0]
        score = hit.score
        prob = hit.probability
        aligned_cols = hit.aligned_cols
    else:
        score, prob, aligned_cols = 0, 0, 0
    coverage = aligned_cols / result.query_length if result.query_length > 0 else 0
    return score, prob, aligned_cols, coverage


def collect_results(work_dir: Path, tasks: list) -> Path:
    """Collect results from completed tasks by downloading and parsing HHR files."""
    results_file = work_dir / "results.tsv"
    
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    results = []
    for task_info in tasks:
        query = task_info["query"]
        target = task_info["target"]
        
        # Download and parse HHR
        hhr_blob = bucket.blob(f"work/hhr/{query}_vs_{target}.hhr")
        try:
            hhr_content = hhr_blob.download_as_text()
            score, prob, aligned_cols, coverage = parse_hhr_content(hhr_content)
            results.append(f"{query}\t{target}\t{score}\t{prob}\t{aligned_cols}\t{coverage:.4f}")
        except Exception as e:
            print(f"  Warning: Could not get results for {query} vs {target}: {e}")
            results.append(f"{query}\t{target}\t0.0\t0.0\t0\t0.0000")
    
    with open(results_file, "w") as f:
        f.write("# query\ttarget\tscore\tprobability\taligned_cols\tcoverage\n")
        f.write("\n".join(results) + "\n")
    
    print(f"  Collected {len(results)} results")
    return results_file


def main():
    if len(sys.argv) < 3:
        print("Usage: python executor_gcp.py <input_dir> <output_dir> [--depth-threshold N]")
        print("Example: python executor_gcp.py ./msa ./results")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    
    # Parse optional arguments
    extra_args = sys.argv[3:]
    
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    script_dir = Path(__file__).parent
    
    print(f"=" * 60)
    print(f"Profalign Google Cloud Batch Executor v1.2")
    print(f"=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Bucket: gs://{BUCKET_NAME}")
    print(f"Docker: {DOCKER_IMAGE}")
    print()
    
    # Step 1: Initialize (prepares profiles with filter + consensus)
    print("[1] Initializing pipeline...")
    print("    (prof_align_cluster.py init prepares profiles: filter + consensus)")
    rc, stdout, stderr = run_command(
        ["python", "prof_align_cluster.py", "init", str(input_dir), str(work_dir)] + extra_args,
        cwd=str(script_dir)
    )
    if rc != 0:
        print(f"Error: {stderr}")
        sys.exit(1)
    
    # Upload prepared profiles to GCS
    # profiles/ contains filtered + consensus versions for hhalign
    print("  Uploading prepared profiles to GCS...")
    upload_directory_to_gcs(work_dir / "profiles", BUCKET_NAME, "work/profiles")
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"[Iteration {iteration}]")
        print(f"{'='*60}")
        
        # Generate queue
        print("  Generating task queue...")
        rc, stdout, stderr = run_command(
            ["python", "prof_align_cluster.py", "gen-queue", str(work_dir)],
            cwd=str(script_dir)
        )
        
        if rc != 0:
            print("  Converged! No more tasks to run.")
            break
        
        # Read queue
        queue_file = work_dir / "current_queue.tsv"
        with open(queue_file) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        
        num_tasks = len(lines)
        print(f"  Tasks to run: {num_tasks}")
        
        if num_tasks == 0:
            print("  No tasks - converged!")
            break
        
        # Prepare task files for batch (0-indexed for BATCH_TASK_INDEX)
        batch_tasks_dir = work_dir / "batch_tasks"
        batch_tasks_dir.mkdir(exist_ok=True)
        
        tasks = []
        for i, line in enumerate(lines):
            parts = line.split("\t")
            task_id, query, target, query_path, target_path = parts
            
            # Convert absolute local paths to GCS-relative paths
            # profiles/ and merged/ contain prepared versions
            query_gcs = f"work/profiles/{Path(query_path).name}"
            target_gcs = f"work/profiles/{Path(target_path).name}"
            
            # Check if it's a merged profile
            if "merged" in query_path:
                query_gcs = f"work/merged/{Path(query_path).name}"
            if "merged" in target_path:
                target_gcs = f"work/merged/{Path(target_path).name}"
            
            task_info = {
                "task_id": i,
                "query": query,
                "target": target,
                "query_path": query_gcs,
                "target_path": target_gcs
            }
            tasks.append(task_info)
            
            # Write task file (0-indexed)
            task_file = batch_tasks_dir / f"task_{i}.json"
            with open(task_file, "w") as f:
                json.dump(task_info, f)
        
        # Upload task definitions
        print("  Uploading task definitions...")
        upload_directory_to_gcs(batch_tasks_dir, BUCKET_NAME, "work/batch_tasks")
        
        # Upload any new merged profiles from previous iteration
        merged_dir = work_dir / "merged"
        if merged_dir.exists() and any(merged_dir.iterdir()):
            print("  Uploading merged profiles...")
            upload_directory_to_gcs(merged_dir, BUCKET_NAME, "work/merged")
        
        # Create and submit batch job
        job_id = f"profalign-iter{iteration}-{int(time.time())}"
        print(f"  Submitting batch job: {job_id}")
        
        job = create_batch_job(job_id, tasks)
        print(f"  Job created: {job.name}")
        
        # Wait for completion
        success = wait_for_job(job.name)
        if not success:
            print("  Job failed! Check Cloud Console for details.")
            print(f"  gcloud batch jobs describe {job_id} --location={REGION}")
            sys.exit(1)
        
        # Collect results from HHR files
        print("  Collecting results from HHR files...")
        results_file = collect_results(work_dir, tasks)
        
        # Download HHR files for local storage (optional, for debugging)
        hhr_dir = work_dir / "hhr"
        hhr_dir.mkdir(exist_ok=True)
        print("  Downloading HHR files...")
        download_directory_from_gcs(BUCKET_NAME, "work/hhr", hhr_dir)
        
        # Process results (performs merges locally using raw profiles)
        print("  Processing results (merges performed locally)...")
        rc, stdout, stderr = run_command(
            ["python", "prof_align_cluster.py", "process-results", str(work_dir), str(results_file)],
            cwd=str(script_dir)
        )
        if rc != 0:
            print(f"Error: {stderr}")
            sys.exit(1)
        
        # Show status
        rc, stdout, stderr = run_command(
            ["python", "prof_align_cluster.py", "status", str(work_dir)],
            cwd=str(script_dir)
        )
        print(stdout)
        
        # Check convergence
        state_file = work_dir / "state.json"
        with open(state_file) as f:
            state = json.load(f)
        
        if state["status"] == "converged":
            print("  Pipeline converged!")
            break
    
    # Finalize
    print(f"\n{'='*60}")
    print("[Finalize] Generating output...")
    rc, stdout, stderr = run_command(
        ["python", "prof_align_cluster.py", "finalize", str(work_dir), str(output_dir)],
        cwd=str(script_dir)
    )
    
    # Print summary
    summary_file = output_dir / "summary.txt"
    if summary_file.exists():
        print()
        print(summary_file.read_text())
    
    print(f"{'='*60}")
    print("Done!")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()
