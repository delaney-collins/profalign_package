"""
task_queue.py - Task queue for pairwise profile comparisons

This module provides:
1. Task definition for pairwise hhalign
2. Queue generation (what pairs need to be compared)
3. Sequential executor (local machine)
4. Result collection into ScoreStore

Design for future parallelization:
- Tasks are independent (can run in any order)
- Each task writes to separate output file
- Results merged after all tasks complete
"""

import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import tempfile
import shutil

logger = logging.getLogger(__name__)


@dataclass
class PairwiseTask:
    """
    Single pairwise comparison task.
    
    Attributes:
        query_profile: Path to query MSA file
        target_profile: Path to target MSA file
        query_name: Name of query profile
        target_name: Name of target profile
        output_hhr: Path for output HHR file
    """
    query_profile: Path
    target_profile: Path
    query_name: str
    target_name: str
    output_hhr: Path
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for serialization."""
        return {
            'query_profile': str(self.query_profile),
            'target_profile': str(self.target_profile),
            'query_name': self.query_name,
            'target_name': self.target_name,
            'output_hhr': str(self.output_hhr),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'PairwiseTask':
        """Create from dictionary."""
        return cls(
            query_profile=Path(d['query_profile']),
            target_profile=Path(d['target_profile']),
            query_name=d['query_name'],
            target_name=d['target_name'],
            output_hhr=Path(d['output_hhr']),
        )


@dataclass
class TaskResult:
    """Result of a single pairwise comparison."""
    query_name: str
    target_name: str
    score: float          # Raw HHsearch score (for Yuri-style distance calculation)
    probability: float    # HHsearch probability (0-100)
    aligned_cols: int
    coverage: float       # Min coverage of query/template (0-1)
    success: bool
    error_message: str = ""
    
    # Computed later after normalization/symmetrization:
    distance: float = 0.0


class TaskQueue:
    """
    Queue of pairwise comparison tasks.
    
    Features:
    - Generate tasks from profile pairs
    - Save/load queue to file (for cluster submission)
    - Track completed tasks
    """
    
    def __init__(self):
        self.tasks: List[PairwiseTask] = []
        self.completed: set = set()  # Set of (query_name, target_name) tuples
    
    def add_task(self, task: PairwiseTask):
        """Add a task to the queue."""
        self.tasks.append(task)
    
    def add_tasks_from_pairs(self,
                             pairs: List[Tuple[str, str]],
                             profile_paths: Dict[str, Path],
                             output_dir: Path):
        """
        Generate tasks from list of profile pairs.
        
        Args:
            pairs: List of (name1, name2) pairs to compare
            profile_paths: Dictionary mapping profile name to MSA file path
            output_dir: Directory for output HHR files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name1, name2 in pairs:
            if name1 not in profile_paths:
                logger.warning(f"Profile not found: {name1}")
                continue
            if name2 not in profile_paths:
                logger.warning(f"Profile not found: {name2}")
                continue
            
            # Create task (use sorted names for consistent file naming)
            sorted_names = sorted([name1, name2])
            output_file = output_dir / f"{sorted_names[0]}_vs_{sorted_names[1]}.hhr"
            
            task = PairwiseTask(
                query_profile=profile_paths[name1],
                target_profile=profile_paths[name2],
                query_name=name1,
                target_name=name2,
                output_hhr=output_file,
            )
            self.tasks.append(task)
    
    def get_pending_tasks(self) -> List[PairwiseTask]:
        """Get tasks that haven't been completed yet."""
        pending = []
        for task in self.tasks:
            key = tuple(sorted([task.query_name, task.target_name]))
            if key not in self.completed:
                pending.append(task)
        return pending
    
    def mark_completed(self, query_name: str, target_name: str):
        """Mark a task as completed."""
        key = tuple(sorted([query_name, target_name]))
        self.completed.add(key)
    
    def save_to_file(self, filepath: Path):
        """Save queue to TSV file (for cluster submission)."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("# query_profile\ttarget_profile\tquery_name\ttarget_name\toutput_hhr\n")
            for task in self.tasks:
                f.write(f"{task.query_profile}\t{task.target_profile}\t"
                       f"{task.query_name}\t{task.target_name}\t{task.output_hhr}\n")
        
        logger.info(f"Saved {len(self.tasks)} tasks to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'TaskQueue':
        """Load queue from TSV file."""
        queue = cls()
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 5:
                    task = PairwiseTask(
                        query_profile=Path(parts[0]),
                        target_profile=Path(parts[1]),
                        query_name=parts[2],
                        target_name=parts[3],
                        output_hhr=Path(parts[4]),
                    )
                    queue.tasks.append(task)
        
        logger.info(f"Loaded {len(queue.tasks)} tasks from {filepath}")
        return queue
    
    def __len__(self):
        return len(self.tasks)


def run_single_task(task: PairwiseTask,
                    add_consensus: bool = True,
                    max_distance: float = 12.0,
                    min_coverage: float = 0.0,
                    gap_threshold: float = 1.0) -> TaskResult:
    """
    Execute a single pairwise comparison task.
    
    Uses hhalign to compare two profiles and extracts distance.
    
    Gap filter (Yuri's sr_filter -gcon):
    Before running hhalign, columns with gap_fraction > gap_threshold are removed.
    This improves distance calculation by focusing on well-aligned regions.
    Yuri uses 0.5 (removes columns with >50% gaps).
    Default 1.0 means no filtering.
    
    Coverage filter (Yuri's -l parameter):
    If min(query_coverage, template_coverage) < min_coverage,
    the pair is considered unrelated and gets max_distance.
    
    Args:
        task: The task to execute
        add_consensus: Add consensus sequence before alignment
        max_distance: Distance to assign when no significant hit
        min_coverage: Minimum coverage threshold (0-1), default 0.0 (no filter)
                     Yuri uses 0.667 as default
        gap_threshold: Max gap fraction for column filtering before hhalign (0-1)
                      Default 1.0 (no filtering). Yuri uses 0.5.
        
    Returns:
        TaskResult with distance and metadata
    """
    from .hhsuite.hhsuite import hhalign
    from .io.formats import read_fasta, write_fasta
    from .alignment.merge import add_consensus_to_alignment, filter_alignment_columns, extract_filtered_sequences
    from .hhsuite.hhr_parser import parse_hhr_file
    
    try:
        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Read query alignment
            query_ali = read_fasta(task.query_profile)
            
            # Apply gap filtering if threshold < 1.0
            if gap_threshold < 1.0:
                filtered_query = filter_alignment_columns(query_ali, gap_threshold=gap_threshold)
                query_ali = extract_filtered_sequences(filtered_query)
            
            # Add consensus
            if add_consensus:
                query_ali = add_consensus_to_alignment(query_ali, f"{task.query_name}@_cons")
            
            query_file = tmpdir / "query.a3m"
            write_fasta(query_ali, query_file)
            
            # Read target alignment
            target_ali = read_fasta(task.target_profile)
            
            # Apply gap filtering if threshold < 1.0
            if gap_threshold < 1.0:
                filtered_target = filter_alignment_columns(target_ali, gap_threshold=gap_threshold)
                target_ali = extract_filtered_sequences(filtered_target)
            
            # Add consensus
            if add_consensus:
                target_ali = add_consensus_to_alignment(target_ali, f"{task.target_name}@_cons")
            
            target_file = tmpdir / "target.a3m"
            write_fasta(target_ali, target_file)
            
            # Run hhalign
            hhr_file = tmpdir / "result.hhr"
            hhalign(query_file, target_file, hhr_file)
            
            # Copy HHR to output location
            task.output_hhr.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hhr_file, task.output_hhr)
            
            # Parse result
            result = parse_hhr_file(hhr_file)
            
            # Extract best hit (should be the only one for pairwise)
            if result.hits:
                hit = result.hits[0]
                score = hit.score          # Raw HMM score (Yuri uses this!)
                probability = hit.probability
                aligned_cols = hit.aligned_cols
                
                # Calculate coverage as min of query and template coverage
                # This matches Yuri's approach: both profiles must be well-covered
                query_len = result.query_length if result.query_length > 0 else 1
                template_len = hit.template_length if hit.template_length > 0 else 1
                
                query_coverage = aligned_cols / query_len
                template_coverage = aligned_cols / template_len
                coverage = min(query_coverage, template_coverage)
                
                # Apply coverage filter - if too low, zero out score
                # Distance calculation happens later with normalization
                if coverage < min_coverage:
                    logger.debug(f"  Low coverage ({coverage:.3f} < {min_coverage}): "
                                f"{task.query_name} vs {task.target_name}")
                    score = 0.0  # Will result in max_distance after normalization
            else:
                score = 0.0
                probability = 0.0
                aligned_cols = 0
                coverage = 0.0
            
            return TaskResult(
                query_name=task.query_name,
                target_name=task.target_name,
                score=score,
                probability=probability,
                aligned_cols=aligned_cols,
                coverage=coverage,
                success=True,
                # distance will be computed later after normalization
            )
    
    except Exception as e:
        logger.error(f"Task failed: {task.query_name} vs {task.target_name}: {e}")
        return TaskResult(
            query_name=task.query_name,
            target_name=task.target_name,
            score=0.0,
            probability=0.0,
            aligned_cols=0,
            coverage=0.0,
            success=False,
            error_message=str(e),
        )


def run_task_queue_sequential(queue: TaskQueue,
                              add_consensus: bool = True,
                              max_distance: float = 12.0,
                              min_coverage: float = 0.0,
                              gap_threshold: float = 1.0,
                              progress_callback=None) -> List[TaskResult]:
    """
    Execute all tasks in queue sequentially.
    
    Args:
        queue: TaskQueue with tasks to execute
        add_consensus: Add consensus before alignment
        max_distance: Max distance for failed comparisons
        min_coverage: Minimum coverage threshold (0-1), default 0.0
                     Yuri uses 0.667 (-l parameter)
        gap_threshold: Max gap fraction for column filtering before hhalign (0-1)
                      Default 1.0 (no filtering). Yuri uses 0.5 (-gcon parameter)
        progress_callback: Optional callback(current, total) for progress
        
    Returns:
        List of TaskResult objects
    """
    results = []
    pending = queue.get_pending_tasks()
    total = len(pending)
    
    logger.info(f"Running {total} pairwise comparisons sequentially...")
    
    for i, task in enumerate(pending):
        if progress_callback:
            progress_callback(i + 1, total)
        else:
            logger.info(f"  [{i+1}/{total}] {task.query_name} vs {task.target_name}")
        
        result = run_single_task(task, add_consensus, max_distance, min_coverage, gap_threshold)
        results.append(result)
        
        queue.mark_completed(task.query_name, task.target_name)
    
    # Summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"Completed: {successful}/{total} successful")
    
    return results


def results_to_score_store(results: List[TaskResult], store=None):
    """
    Add task results to a ScoreStore.
    
    Stores raw scores from TaskResult. Distance matrix is computed later
    using normalization and symmetrization (Yuri's algorithm).
    
    Args:
        results: List of TaskResult objects
        store: Existing ScoreStore (or None to create new)
        
    Returns:
        ScoreStore with results added
    """
    from .pipeline.pipeline_utils import ScoreStore
    
    if store is None:
        store = ScoreStore()
    
    for r in results:
        if r.success:
            store.add(
                r.query_name,
                r.target_name,
                r.score,  # Raw score, NOT distance!
                r.probability,
                r.aligned_cols,
                r.coverage,
                append_to_file=True
            )
    
    return store


# Alias for backward compatibility
results_to_distance_store = results_to_score_store
