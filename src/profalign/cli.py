#!/usr/bin/env python3
"""
Unified CLI for the profalign package.

Provides:
    profalign run ...
    profalign cluster init ...
    profalign cluster gen-queue ...
    profalign cluster process-results ...
    profalign cluster status ...
    profalign cluster finalize ...
    profalign precluster ...

Example:
    profalign run examples/msa results/
    profalign cluster init examples/msa work/
    profalign precluster sequences.fasta precluster_out/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from .version import __version__

# Main pipeline
from .profalign_pipeline import PipelineConfig, ProfileAlignmentPipeline

# Cluster/executor mode
from .cluster import (
    cmd_finalize,
    cmd_gen_queue,
    cmd_init,
    cmd_process_results,
    cmd_status,
)

# Preclustering
from .precluster import (
    ClusterConfig,
    check_dependencies,
    run_precluster,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        root.setLevel(level)
        for handler in root.handlers:
            handler.setLevel(level)


# ---------------------------------------------------------------------
# Shared parser helpers
# ---------------------------------------------------------------------

def add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing input MSA files (FASTA format).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for results.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of iterations (default: 10).",
    )
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=1.2,
        help="Tree depth threshold for subtree cutting (default: 1.2).",
    )
    parser.add_argument(
        "--depth-weight-mode",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help=(
            "Weighting mode for tree depth calculation (default: 2). "
            "0=by number of leaves, 1=by total branch length, 2=unweighted."
        ),
    )
    parser.add_argument(
        "--merge-gap-threshold",
        type=float,
        default=0.667,
        help="Gap threshold for column filtering during merge (default: 0.667).",
    )
    parser.add_argument(
        "--distance-gap-threshold",
        type=float,
        default=0.5,
        help=(
            "Gap threshold for column filtering before distance calculation "
            "(default: 0.5). Use 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--length-coverage",
        type=float,
        default=0.667,
        help=(
            "Minimum alignment coverage for distance calculation "
            "(default: 0.667)."
        ),
    )
    parser.add_argument(
        "--no-consensus",
        action="store_true",
        help="Do not add consensus sequences for hhalign.",
    )
    parser.add_argument(
        "--min-profiles",
        type=int,
        default=1,
        help="Stop when this many profiles remain (default: 1).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )


def add_common_cluster_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with MSA files.",
    )
    parser.add_argument(
        "work_dir",
        type=Path,
        help="Work directory.",
    )
    parser.add_argument("--depth-threshold", type=float, default=1.2)
    parser.add_argument(
        "--depth-weight-mode",
        type=int,
        default=2,
        choices=[0, 1, 2],
    )
    parser.add_argument("--merge-gap-threshold", type=float, default=0.667)
    parser.add_argument("--distance-gap-threshold", type=float, default=0.5)
    parser.add_argument(
        "--length-coverage",
        type=float,
        default=0.667,
        help="Minimum alignment coverage for distance calculation.",
    )
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )


def add_common_precluster_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input_fasta",
        type=Path,
        help="Input FASTA file with sequences.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory.",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.5,
        help="Sequence identity threshold (default: 0.5).",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.333,
        help="Coverage threshold (default: 0.333).",
    )
    parser.add_argument(
        "--cov-mode",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Coverage mode: 0=longer, 1=target, 2=query (default: 1).",
    )
    parser.add_argument(
        "--cluster-mode",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Cluster mode: 0=set-cover, 1=connected, 2=greedy (default: 2).",
    )
    parser.add_argument(
        "--aligner",
        choices=["muscle", "mafft"],
        default="muscle",
        help="MSA tool (default: muscle).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads (default: 4).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help="Minimum cluster size to output (default: 1).",
    )
    parser.add_argument(
        "--no-singletons",
        action="store_true",
        help="Skip singleton clusters.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )


# ---------------------------------------------------------------------
# Command runners
# ---------------------------------------------------------------------

def run_pipeline_command(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)

    if not args.input_dir.exists():
        logger.error("Input directory does not exist: %s", args.input_dir)
        return 1

    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        depth_threshold=args.depth_threshold,
        depth_weight_mode=args.depth_weight_mode,
        merge_gap_threshold=args.merge_gap_threshold,
        distance_gap_threshold=args.distance_gap_threshold,
        length_coverage=args.length_coverage,
        add_consensus=not args.no_consensus,
        min_profiles_to_stop=args.min_profiles,
    )

    pipeline = ProfileAlignmentPipeline(config)

    try:
        pipeline.run()
        return 0
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


def run_cluster_command(args: argparse.Namespace) -> int:
    configure_logging(getattr(args, "verbose", False))

    commands = {
        "init": cmd_init,
        "gen-queue": cmd_gen_queue,
        "process-results": cmd_process_results,
        "status": cmd_status,
        "finalize": cmd_finalize,
    }

    handler = commands.get(args.cluster_command)
    if handler is None:
        logger.error("Unknown cluster subcommand: %s", args.cluster_command)
        return 1

    # Your existing cluster functions expect argparse.Namespace and should work as-is.
    try:
        return handler(args)
    except Exception as exc:
        logger.exception("Cluster command failed: %s", exc)
        return 1


def run_precluster_command(args: argparse.Namespace) -> int:
    configure_logging(args.verbose)

    logger.info("Checking dependencies...")
    deps = check_dependencies()

    if not deps["mmseqs"]:
        logger.error("mmseqs2 not found. Install with: conda install -c bioconda mmseqs2")
        return 1

    if args.aligner == "muscle" and not deps["muscle"]:
        logger.error("muscle not found. Install with: conda install -c bioconda muscle")
        return 1

    if args.aligner == "mafft" and not deps["mafft"]:
        logger.error("mafft not found. Install with: conda install -c bioconda mafft")
        return 1

    if not args.input_fasta.exists():
        logger.error("Input file not found: %s", args.input_fasta)
        return 1

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
        keep_singletons=not args.no_singletons,
    )

    try:
        _, stats = run_precluster(config)

        logger.info("")
        logger.info("=" * 50)
        logger.info("Pre-clustering complete!")
        logger.info("  Input sequences: %s", stats["input_sequences"])
        logger.info("  Clusters: %s", stats["num_clusters"])
        logger.info("  Singletons: %s", stats["singletons"])
        logger.info("  Largest cluster: %s", stats["largest_cluster"])
        logger.info("  Output MSA files: %s", stats["output_files"])
        logger.info("")
        logger.info("MSA files written to: %s", config.output_dir / "msa")
        logger.info("")
        logger.info("Next step:")
        logger.info("  profalign run %s <results_dir>", config.output_dir / "msa")
        return 0

    except Exception as exc:
        logger.exception("Pre-clustering failed: %s", exc)
        return 1


# ---------------------------------------------------------------------
# Parser builder
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profalign",
        description="Unified command-line interface for the profalign package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  profalign run examples/msa results/
  profalign run examples/msa results/ --depth-threshold 1.0 --max-iterations 5

  profalign cluster init examples/msa work/
  profalign cluster gen-queue work/
  profalign cluster process-results work/ results.tsv
  profalign cluster status work/
  profalign cluster finalize work/ final_output/

  profalign precluster sequences.fasta precluster_out/
  profalign precluster sequences.fasta precluster_out/ --identity 0.7
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # -----------------------------------------------------------------
    # run
    # -----------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run the main iterative profile-alignment pipeline.",
    )
    add_common_run_arguments(run_parser)
    run_parser.set_defaults(func=run_pipeline_command)

    # -----------------------------------------------------------------
    # cluster
    # -----------------------------------------------------------------
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster/executor mode commands for external job orchestration.",
    )
    cluster_subparsers = cluster_parser.add_subparsers(dest="cluster_command")

    cluster_init = cluster_subparsers.add_parser(
        "init",
        help="Initialize cluster-mode work directory.",
    )
    add_common_cluster_init_arguments(cluster_init)
    cluster_init.set_defaults(func=run_cluster_command)

    cluster_queue = cluster_subparsers.add_parser(
        "gen-queue",
        help="Generate task queue for current iteration.",
    )
    cluster_queue.add_argument("work_dir", type=Path, help="Work directory.")
    cluster_queue.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    cluster_queue.set_defaults(func=run_cluster_command)

    cluster_results = cluster_subparsers.add_parser(
        "process-results",
        help="Process hhalign results and perform merges.",
    )
    cluster_results.add_argument("work_dir", type=Path, help="Work directory.")
    cluster_results.add_argument(
        "results_file",
        type=Path,
        nargs="?",
        help="Optional results TSV file.",
    )
    cluster_results.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    cluster_results.set_defaults(func=run_cluster_command)

    cluster_status = cluster_subparsers.add_parser(
        "status",
        help="Show current cluster-mode pipeline status.",
    )
    cluster_status.add_argument("work_dir", type=Path, help="Work directory.")
    cluster_status.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    cluster_status.set_defaults(func=run_cluster_command)

    cluster_finalize = cluster_subparsers.add_parser(
        "finalize",
        help="Generate final output from cluster-mode work directory.",
    )
    cluster_finalize.add_argument("work_dir", type=Path, help="Work directory.")
    cluster_finalize.add_argument("output_dir", type=Path, help="Output directory.")
    cluster_finalize.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    cluster_finalize.set_defaults(func=run_cluster_command)

    # -----------------------------------------------------------------
    # precluster
    # -----------------------------------------------------------------
    precluster_parser = subparsers.add_parser(
        "precluster",
        help="Pre-cluster raw sequences before profile alignment.",
    )
    add_common_precluster_arguments(precluster_parser)
    precluster_parser.set_defaults(func=run_precluster_command)

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    if args.command == "cluster" and not getattr(args, "cluster_command", None):
        parser.parse_args([args.command, "--help"])
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())