# Prospective CSV composition evaluation

## Protocol

The protocol was frozen before execution using seed 20261113.
Model, lookup, teacher, and catalog fingerprints were checked.

The test contains:
- 12 ordered compositions of three familiar operations.
- All 48 base configurations per composition: 576 environments.
- 48 known-configuration controls.
- 100 newly generated evaluation tables.

Budgets remain eight calls, 24 cells, and 512 UTF-8 bytes.
All three prespecified SSMs are evaluated without parameter updates.

This is a prospective test within the same synthetic domain, not an
independent blind evaluation or discovery of new transformation primitives.

## Results

| Policy | Detected / episodes | Detection rate | Mean calls | Mean rollout microseconds |
|---|---:|---:|---:|---:|
| fixed | 216/576 | 37.50% | 5.625 | 123.2 |
| random | 2406/2880 | 83.54% | 3.665 | 75.5 |
| lookup | 576/576 | 100.00% | 1.583 | 49.2 |
| symbolic | 576/576 | 100.00% | 1.583 | 5120.8 |
| ssm_97001 | 576/576 | 100.00% | 1.583 | 779.1 |
| ssm_97002 | 576/576 | 100.00% | 1.583 | 676.0 |
| ssm_97003 | 576/576 | 100.00% | 1.583 | 660.2 |

The lookup, teacher, and each SSM detect all 576 composed environments.
They correctly recognize all 48 known configurations without false rejection.

For random selection, five paths are evaluated per environment.
Its 2,880 composed episodes are not 2,880 independent environments.

## What transfers

The composed behaviors are new, but lookup, teacher, and SSMs encounter
no decision states absent from the branched training-state table.

The result demonstrates reuse of a pretrained testing policy on these
new compositions. It does not demonstrate generalization to new decision
states, interpretation of arbitrary CSV semantics, or reproduction of
unknown tool behavior.

The evaluation tables provide witnesses distinguishing every tested
composition from the known family. Matching on a finite table set would
not by itself establish global behavioral equivalence.

## Practical comparison

Lookup remains the efficient reference: approximately 49 microseconds per
composed episode, versus 660-779 microseconds for the SSMs.

Teacher timing reflects an uncached implementation.
Timings include selection and known-family verification, but exclude
loading, simulator precomputation, and final scoring.
They are indicative measurements from this run, not a universal benchmark.

No SSM advantage over lookup is demonstrated.

## Metadata erratum

The raw result retains an inherited value of 192 in
config.reserved_composition_environments.

The actual data contains 12 x 48 = 576 distinct composed environments.
This accounting error does not change scores or evaluated cases.

The raw file is preserved. The companion summary records the correction
and its source hash. The evaluator code now computes the count dynamically.

## Limitations and decision

- Familiar programmed operations and catalog.
- Handcrafted simulator and symbolic observation features.
- Related configurations within each composition.
- All three seeds share evaluation environments.
- No new decision states for the successful adaptive policies.
- Detection is distinct from learning to reproduce the tool.

Freeze this result without retuning on it. The compositions are now
consulted and must be treated as regression cases in future work.

## Artifacts

The frozen protocol is under autonomous_lab/data/csv_prospective_01/.
Detailed local results remain under autonomous_lab/runs/csv_prospective_01/,
excluded from Git.
