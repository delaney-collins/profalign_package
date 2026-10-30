# Profalign

Version 2.3.2



Iterative profile-profile alignment and merging pipeline for protein family analysis.



This package is a Python reimplementation of Yuri Wolf's prof\_align pipeline, which uses the HH-suite profile alignment to iteratively merge related multiple sequence alignments (MSAs) into larger profiles.



### Installation



1\. Create a Python environment

Using conda or mamba is recommended:



conda create -n profalign python=3.11

conda activate profalign



2\. Install external dependencies

The pipeline requires HH-suite (specifically hhalign) to be available in PATH:



conda install -c bioconda -c conda-forge hhsuite



Optional dependencies for preclustering raw sequences:



conda install -c bioconda mmseqs2 muscle



3\. Install profalign



pip install profalign



This installs the package and exposes the CLI command.



### Quick Start

Run the included example dataset:



profalign run examples/msa results



Expected runtime: \~15-20 minutes.



Check results:



cat results/summary.txt



Example output:



Final profiles: 14



You can inspect the merge history:



cat results/work/merge\_history.tsv



Clean up:



rm -rf results



### Command Line Interface

The package provides a unified CLI:



profalign <command> \[options]



Available commands:



profalign run

profalign cluster

profalign precluster



### Running the Pipeline

The simplest mode runs everything locally:



profalign run INPUT\_DIR OUTPUT\_DIR



Example:



profalign run examples/msa results



Verbose logging:



profalign run examples/msa results -v



With parameter changes:



profalign run examples/msa results \\

&#x09;--depth-threshold 2.0 \\

&#x09;--length-coverage 0.8



### Starting from Raw Sequences

If you have unaligned sequences, first cluster and align them using the precluster command:



profalign precluster sequences.fasta preclusteres/



This produces MSAs in preclustered/msa/



Then run the main pipeline:



profalign run preclustered/msa results



### Cluster/HPC Mode

For large datasets, pairwise HHalign comparisons can be executed on a cluster.



Workflow:



profalign cluster init

profalign cluster gen-queue

(run tasks on HPC)

profalign cluster process-results

profalign cluster finalize



Example:



profalign cluster init examples/msa work

profalign cluster gen-queue work



Submit tasks to your scheduler.



After completion:



profalign cluster process-results work results.tsv



Repeat until convergence, then finalize:



profalign cluster finalize work results



### Example Dataset

The repository includes a small dataset:

examples/msa/

This contains 30 CRISPR protein MSAs used for testing.


This pipeline:

1. Calculates pairwise HHalign scores
2. Converts scores to distances
3. Builds a UPGMA tree
4. Extracts subtrees
5. Merges related profiles
6. Repeats until convergence



### Output Structure

results/

├── summary.txt

├── pipeline.log

└── work/

&#x20;   ├── profiles/

&#x20;   ├── profiles\_raw/

&#x20;   ├── merged/

&#x20;   ├── merged\_raw/

&#x20;   ├── distances.tsv

&#x20;   ├── merge\_history.tsv

&#x20;   └── hhr/



Important files:



| File                | Description            |

| ------------------- | ---------------------- |

| `summary.txt`       | Human-readable summary |

| `merge\_history.tsv` | Full merge genealogy   |

| `distances.tsv`     | Cached HHalign scores  |



### Parameters

| Parameter                  | Default | Description                  |

| -------------------------- | ------- | ---------------------------- |

| `--depth-threshold`        | 1.2     | subtree extraction threshold |

| `--depth-weight-mode`      | 2       | tree depth weighting         |

| `--length-coverage`        | 0.667   | minimum alignment coverage   |

| `--distance-gap-threshold` | 0.5     | gap filtering before scoring |

| `--merge-gap-threshold`    | 0.667   | gap filtering during merge   |

| `--max-iterations`         | 10      | maximum merge iterations     |

| `--min-profiles`           | 1       | stop condition               |



Key parameter: --depth-threshold

| Value | Behavior                          |

| ----- | --------------------------------- |

| 0.5   | conservative merging              |

| 1.2   | default (matches Yuri’s pipeline) |

| 2.0   | aggressive merging                |



Example:



profalign run examples/msa results --depth-threshold 2.0



### Input Format

Input directory must contain aligned FASTA files:



profiles/

├── family\_A.fasta

├── family\_B.fasta

└── ...



Each file represents one profile.

Files may contain a multiple sequence alignment or a single sequence. If sequences are not aligned, use precluster.



Distance Metric

Distance is derived from HHalign scores using Yuri's algorithm:



score\[i,j]      = HHalign score

normalized\[i,j] = score\[i,j] / score\[i,i]

sym\[i,j]        = max(normalized\[i,j], normalized\[j,i])

distance\[i,j]   = -log(sym\[i,j])



Interpretation:



| Score | Distance | Meaning      |

| ----- | -------- | ------------ |

| 1.0   | 0        | identical    |

| 0.5   | 0.69     | very similar |

| 0.1   | 2.3      | related      |

| 0.01  | 4.6      | distant      |



### Running Tests

Unit test:



pytest



Integration tests (requires HHsuite):

pytest -m integration



Requirements

Core pipeline:

* Python 3.8+
* HH-suite (hhalign)
* numpy
* scipy



Optional (required for preclustering):

* mmseqs2
* muscle or mafft

### 

### Troubleshooting

##### "hhalign not found"

Check installation:



which hhalign

hhalign -h



Install via Bioconda:



conda install -c bioconda hhsuite



##### Pipeline is slow

Distance calculation scales as O(n²)



Suggestions:

* test with smaller datasets first
* use cluster mode
* restart runs (scores are cached)



##### No merges occur

Possible causes:

* profiles not related
* \--depth-threshold is too low
* coverage filter is too strict



try:



profalign run examples/msa results --depth-threshold 2.0



### Project Structure

profalign/

├── pyproject.toml

├── README.md

├── CHANGELOG.md

├── LICENSE

├── src/

│   └── profalign/

│       ├── cli.py

│       ├── profalign\_pipeline.py

│       ├── precluster.py

│       ├── version.py

│       ├── cluster.py

│       ├── alignment/

│       ├── clustering/

│       ├── executors/

│       ├── hhsuite/

│       ├── io/

│       ├── pipeline/

│       └── scripts/

├── examples/

│   └── msa/

│    └── raw\_fasta/

├── tests/

└── docs/



### License

MIT



