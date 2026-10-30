# Profile alignment and merging: Methods

## Algorithm overview

ProfAlign iteratively merges multiple sequence alignments (MSAs) of related
protein families based on their pairwise profile-profile similarity. Starting
from a set of N input MSA profiles, it repeats the following cycle until
convergence:

1. **Pairwise comparison.** All profile pairs are compared using HHalign
   (HH-suite 3.x), which aligns two profile HMMs using the Viterbi algorithm
   and returns a raw alignment score, the number of aligned columns, and
   a probability estimate.

2. **Distance matrix construction.** Raw scores are converted to distances
   following the procedure of hhres2dis_mat (Wolf, unpublished):
   - Normalization: each score is divided by the query's self-score
     (score of the profile aligned against itself), yielding a similarity
     in the range [0, 1].
   - Coverage filter: pairs where the aligned fraction of either profile
     is below a threshold (default 0.667) are assigned maximum distance.
   - Symmetrization: for each pair (i, j), the maximum of the two
     directional normalized scores is taken.
   - Distance conversion: d(i, j) = −log(similarity), capped at 12.0.

3. **UPGMA tree.** A hierarchical clustering tree is built from the distance
   matrix using the UPGMA (average linkage) algorithm via scipy.

4. **Subtree extraction.** The tree is cut into subtrees whose root depth
   (weighted average distance to leaves, unweighted mode) does not exceed
   a threshold (default 1.2). Trivial subtrees (single leaves) are ignored.

5. **Guided merging.** Each nontrivial subtree is merged bottom-up following
   the guide tree topology. At each internal node, two MSAs are merged by:
   - Filtering columns with >66.7% gaps from each MSA (temporary, for
     comparison only).
   - Adding a BLOSUM62-based consensus sequence as the first sequence
     in each filtered MSA (defines match states via HHalign's -M first).
   - Running HHalign in global mode to produce a pairwise alignment with
     an alignment table (-atab).
   - Parsing the alignment table to establish column correspondences.
   - Reconstructing the full merged MSA by reinserting filtered columns
     (N-terminal tail, aligned core with insertions, C-terminal tail).

6. **Convergence.** Steps 1-5 repeat with the updated profile set (merged
   profiles replacing their parents). The process stops when no nontrivial
   subtrees are found, or a maximum iteration count is reached.


## Comparison tool

ProfAlign uses HHalign rather than HHsearch for pairwise profile comparison.
Both tools perform profile HMM alignment using the same Viterbi scoring
function (Söding, 2005; Steinegger et al., 2019). HHsearch is designed for
database search (one query against many targets), while HHalign compares
exactly two profiles. For the symmetric pairwise distance matrix required by
our pipeline, HHalign has two advantages: (1) it applies context-specific
pseudocounts symmetrically to both profiles, whereas HHsearch uses faster
but less accurate substitution-matrix pseudocounts for database targets;
(2) it avoids the prefilter stage of HHsearch, which could discard valid
but low-scoring pairs in small profile sets.


## Parameters

All numeric parameters match the defaults of the original Perl pipeline
(Wolf, unpublished) unless noted:

| Parameter | Value | Origin |
|-----------|-------|--------|
| Score normalization | Query self-score | hhres2dis_mat -n=3 |
| Coverage threshold | 0.667 | hhres2dis_mat -l=0.667 |
| Symmetrization | max(d_ij, d_ji) | hhres2dis_mat -s=1 |
| Distance transform | −log, cap 12.0 | hhres2dis_mat -d=2 |
| Gap filter (comparison) | 0.5 | sr_filter -gcon=0.5 |
| Gap filter (merge) | 0.667 | sr_filter -grcut=0.667 |
| Subtree depth threshold | 1.2 | tree_tips -d=1.2 |
| Depth weight mode | Unweighted (mode 2) | tree_tips -w=2 |


## Implementation

ProfAlign is implemented in Python 3. Hierarchical clustering uses
scipy.cluster.hierarchy. Profile HMM alignment uses HH-suite 3.x
(Steinegger et al., 2019). The pipeline supports three execution modes:
sequential (prof_align.py), local parallel (executor_local.py), and
distributed via Google Cloud Batch or SLURM (prof_align_cluster.py).
Only the pairwise HHalign comparisons are parallelized; tree construction,
subtree extraction, and guided merging are performed sequentially.

Source code is available at [URL].


## Known differences from the original pipeline

The original Perl pipeline (Wolf, unpublished) used HHsearch 1.5.1 for
pairwise comparison and R hclust for UPGMA clustering. ProfAlign uses
HHalign from HH-suite 3.x and scipy linkage. The scoring algorithm
(Viterbi HMM-HMM alignment) is identical between HHsearch and HHalign;
absolute scores may differ slightly due to version-dependent pseudocount
handling. The UPGMA algorithm is mathematically identical between R and
scipy; tree heights are calibrated to match the original pipeline's
behavior (distances are halved before clustering to compensate for
implementation differences in height reporting).

The original pipeline preserved columns with 50-66.7% gaps as insert
states in the profile HMM during comparison, while ProfAlign removes them.
On the test dataset of 30 CRISPR-Cas protein profiles, only 3.3% of
alignment columns fall in this range, making the practical impact minimal.

Consensus sequences are computed using uniform sequence weights with
BLOSUM62 scoring, whereas the original used Henikoff position-based
weights. This affects consensus residue choice only in columns with
two or more competing amino acids at similar frequencies.


## References

Söding J. Protein homology detection by HMM-HMM comparison.
Bioinformatics. 2005;21(7):951-960.

Steinegger M, Meier M, Mirdita M, Vöhringer H, Haunsberger SJ,
Söding J. HH-suite3 for fast remote homology detection and deep
protein annotation. BMC Bioinformatics. 2019;20(1):473.
