# ProfAlign Changelog

## v2.2.2 (2026-02-23)

Bugfix release.

### Bug fix
- **Fixed `prof_align_cluster.py`** — merge counter now counts all merges,
  not just subtrees. Previously reported 13 instead of 16 for test dataset.

### Cleanup
- Removed hardcoded version strings from script docstrings (`precluster.py`,
  `executor_gcp.py`, `executor_local.py`, `prof_align_cluster.py`).
  Version is now read from `VERSION` file only.

## v2.2.1 (2026-02-23)

Bugfix release — 6 issues found during code review and runtime testing.

### Critical fixes (blocked execution)
- **Fixed `prof_align.py`** — `gap_threshold=` → `merge_gap_threshold=` in
  IterationConfig construction (line 212). Pipeline crashed with TypeError
  after completing all pairwise comparisons (~10 min wasted per run).
- **Fixed `lib/hhr_parser.py:parse_hhr_string()`** — `result.query_name` →
  `result.query_id`, `result.match_columns` → `result.query_length`.
  Attributes renamed in v2.2.0 HHR parser consolidation but wrapper not updated.
- **Fixed `parse_hhr.py`** — same attribute renames: `query_name` → `query_id`,
  `hits[0].name` → `hits[0].target_id`, `match_columns` → `query_length`.
- **Fixed `executor_gcp.py`** — `result.match_columns` → `result.query_length`.

### Silent correctness fix
- **Fixed `prof_align_cluster.py`** — added `matrix / 2.0` before
  hierarchical_clustering (line 458). The dd/=2 correction from v2.0.0 was
  applied in iteration.py but never propagated to cluster mode, causing
  cluster mode to be 2× stricter than local mode at the same depth_threshold.

### Parameter fix
- **Fixed `examples/run_hhalign_task.sh`** — added `-M first -id 100 -diff inf`
  to hhalign call. Without these, hhalign uses different match state definition
  than the rest of the pipeline, producing inconsistent scores.

## v2.2.0 (2026-02-22)

Code cleanup release addressing both colleague reviews. No algorithmic changes.

### Bug fixes (post-review)
- **Fixed `executor_gcp.py`** — added missing `sys.path.insert(.../lib)`
  so the consolidated `parse_hhr_string()` import actually works at runtime.
- **Fixed `tests/test_pipeline.py`** — `gap_threshold=` → `merge_gap_threshold=`
  in `IterationConfig` construction (was broken by the rename).
- **Integrated `parse_hhr.py` into `examples/run_hhalign_task.sh`** — replaced
  inline `grep/awk` HHR parser with `python parse_hhr.py` call, completing
  the HHR parser consolidation across the entire repo.

### Import system cleanup
- **Removed all dual try/except imports** from lib/ modules — 6 files,
  ~80 lines of duplicated imports eliminated. All lib/*.py now use bare
  imports (e.g. `from formats import ...`), consistent with scripting style.
- **Centralized sys.path** in `tests/conftest.py` — removed individual
  `sys.path.insert` from all 16 test files.
- **Fixed broken sys.path in tools/** — `debug_ffindex.py`, `diagnose_merge.py`,
  `show_tree.py` pointed to nonexistent `tools/lib/` instead of `../lib/`.
- **Fixed `analyze_distances.py`** — relative sys.path + CRLF line endings.

### HHR parser consolidation
- **`parse_hhr.py`** rewritten as thin wrapper (126 → 59 lines) over
  `lib/hhr_parser.py` instead of maintaining a separate regex parser.
- **`executor_gcp.py`** — removed duplicate `parse_hhr_content()` regex
  parser, replaced with call to `lib/hhr_parser.py:parse_hhr_string()`.
- **Added `parse_hhr_string()`** to `lib/hhr_parser.py` for parsing HHR
  content from strings (needed by GCP executor).

### Dead code removal
- **Moved `lib/provenance.py`** (283 lines) to `tools/` — was not imported
  by any module, entry point, or test. Useful utility, wrong location.
- **Removed deprecated `DistanceStore`/`DistancePair` aliases** from
  `__init__.py` exports. All tests migrated to `ScoreStore`/`ScorePair`.

### Parameter naming
- **Renamed `--gap-threshold` → `--merge-gap-threshold`** in CLI and
  config to avoid confusion with `--distance-gap-threshold`. Both
  colleague reviewers independently flagged this naming collision.
  Affects: `prof_align.py`, `prof_align_cluster.py`, `executor_local.py`,
  `lib/iteration.py` (IterationConfig). Internal function parameters
  in `lib/merge.py` and `lib/guided_align.py` unchanged.

### Metadata
- **`lib/version.py`** now reads from `VERSION` file (single source of truth)
  and is exported via `lib/__init__.py` as `__version__`.
- **`lib/__init__.py`** updated: added `__version__` export, marked
  `ffindex_db` as optional in module docstring.
- Updated README.md, requirements.txt to v2.2.0.


## v2.1.1 (2026-02-20)

Patch release addressing colleague review feedback (test infrastructure,
cluster-mode race condition, cross-platform compatibility, stale docs).

### Test infrastructure
- **Added `tests/conftest.py`** — shared fixtures (`msa_dir`, `has_hhalign`)
  and custom pytest markers (`integration`, `slow`)
- **Marked integration tests** with `@pytest.mark.integration`:
  `test_prof_align.py` (test_small_pipeline, test_convergence) and
  `test_task_queue.py` (test_execution_on_real_data) now require explicit
  `-m integration` flag and skip cleanly when HH-suite is not installed
- All unit tests now pass cleanly with bare `pytest` (no HH-suite needed)

### Bug fixes
- **Fixed merge_tmp race in `prof_align_cluster.py`** — `merge_wrapper()`
  now uses `merge_tmp/<output_stem>/` subdirectory per merge (matching
  `iteration.py` behavior), preventing file collisions in parallel cluster mode
- **Fixed `ffindex_db.check_ffindex_tools()`** — replaced
  `subprocess.run(['which', ...])` with `shutil.which()` for Windows
  compatibility

### Documentation
- Updated README.md version to 2.1.1, fixed Quick Start `unzip` command
  (was referencing v1.0.19)
- Updated requirements.txt header (was v1.0.19), clarified that
  hhmake/hhsearch are used by auxiliary modules (ffindex_db, cluster mode)


## v2.1.0 (2026-02-20)

### Bug fixes
- **Restored `build_score_matrix` in hhr_parser.py** — erroneously removed during
  v2.0.0 dead code cleanup, but still called by `build_distance_matrix`. Caused
  NameError when running test_ffindex (hhsearch-based all-vs-all workflow).

### Test fixes
- Fixed 3 tests missing `from pathlib import Path` import
  (test_coverage_filter, test_merge_tails, test_weighted_depth)
- Fixed test_coverage_filter.py: `DistancePair(distance=...)` →
  `ScorePair(score=...)`, added missing `score` field to TaskResult
- Fixed test_pipeline_utils.py: merge ID length assertion 10 → 14
- Fixed test_iteration.py: missing `calc_tree_weighted_depth` call → all subtrees
  had depth=0 → always returned 1 giant subtree
- Fixed test_pipeline.py: mock distances → mock scores with self-scores + both
  directions (matching what iteration.py expects from ScoreStore)
- Fixed test_prof_align.py: path `../data/msa` → resolved `examples/msa`,
  removed non-existent PipelineConfig params (max_merges_per_iteration, distance_threshold)
- 12/16 tests now pass without external tools

### Documentation
- Updated README.md to v2.1.0
- Added Methods section for preprint (docs/methods.md)
- Corrected CHANGELOG to reflect actual changes per version

### Calibration results (no code changes needed)
- dd/=2: confirmed already fixed in v2.0.0 (matrix / 2.0 in iteration.py)
- Gap column census: 3.3% of columns in 50-66.7% gap range across 30 test
  profiles → insert state column difference vs Yuri is negligible
- hhsearch vs hhalign: documented as conscious design decision
  (hhalign more appropriate for pairwise distance matrix task)


## v2.0.0 (2026-02-19)

### Critical fixes
- **dd/=2 UPGMA height correction** (iteration.py)
  Distance matrix divided by 2 before scipy linkage, matching Yuri's
  run_R_hclust behavior. Without this, ProfAlign was 2x stricter at
  the same depth_threshold.
- **merge_tmp isolation** (iteration.py)
  Each merge uses its own subdirectory (merge_tmp/<merge_id>/).
- **Removed double coverage filter**: task-level min_coverage removed,
  canonical filter in distance_matrix.py retained.
- **Test sys.path**: all 15 tests use consistent resolve()-based path.

### Bug fixes
- Added DistancePair/DistanceStore aliases in pipeline_utils.py
- Renamed guided_align.MergeRecord → GuidedMergeRecord (name collision)
- Fixed test_weighted_depth.py import path

### Dead code removal (−1010 lines total from v1.0.19)
- clustering.py: −246 lines (7 dead functions)
- hhr_parser.py: −196 lines (4 dead functions)
- hhsuite.py: −260 lines (6 dead functions)
- ffindex_db.py: −107 lines (create_hhm_database, create_a3m_database)
- merge.py: −65 lines (1 dead function)
- distance_matrix.py: −48 lines (1 dead function)
- guided_align.py: −19 lines (1 dead function)
- formats.py: −25 lines (fasta_to_seqrows, seqrows_to_fasta)
- provenance.py: −31 lines (get_sequences_by_original_profile)
- __init__.py: −9 lines (updated exports)

### Line counts
- v1.0.19: 5825 lines (lib/)
- v2.0.0:  4821 lines (lib/)


## v1.0.19

- Fixed lib/__init__.py broken imports
- Updated merge_id from 8 to 12 hex chars


## v1.0.18

- Initial audited version
- version.py showed 1.0.7 (desynchronized)
