# SSM-guided rule ranking: search reduction versus runtime

## Question

Can a trained SSM accelerate symbolic hypothesis search by ranking
candidates before an independent consistency verifier examines them?

## Protocol

The programmed family contains 137 Boolean functions on five input bits.
All rules appear in training. Observation contexts are split into
120 training contexts and 30 validation contexts.

Training targets distribute probability uniformly across every rule
compatible with the observations. The SSM learns from symbolic supervision;
it does not invent the hypothesis family.

Three models use seeds 91011, 91012, and 91013.

At evaluation, both methods stop at the first candidate consistent with
every observation. Compatibility does not establish the true rule.

## Ranking results

Means across three seeds:

| Observations | Fixed candidates checked | SSM candidates checked | Fixed unknown-output accuracy | SSM unknown-output accuracy |
|---|---:|---:|---:|---:|
| 4 | 28.17 | 2.18 | 65.70% | 63.79% |
| 8 | 56.34 | 9.29 | 85.23% | 85.10% |
| 12 | 65.71 | 14.03 | 94.76% | 94.73% |

The SSM reduces candidate checks consistently. It does not improve
prediction accuracy; with four observations it performs worse.

## CPU timing

The benchmark uses 300 validation episodes, equally divided among
4, 8, and 12 observations. It runs one episode at a time, with one CPU
thread, warmup, and five timing repetitions.

The following values average the per-model median latencies:

| Method | Microseconds per episode |
|---|---:|
| Symbolic Python search | 27.93 |
| Symbolic vectorized verification | 38.33 |
| SSM inference + sorting + verification | 580.83 |

The guided method is approximately 20.8 times slower than Python
search in this benchmark.

Neural timing includes inference, sorting, and candidate verification.
All timings exclude file loading, model loading, and input preparation.

## Interpretation

Candidate-check reduction is not runtime acceleration. On this small
family, inference overhead outweighs the saved verification work.

Vectorization also costs more than the short Python search here.
This does not establish the same ranking on larger candidate spaces,
other hardware, or expensive verification procedures.

## Limitations

- Programmed family and synthetic training labels.
- Familiar rules, with held-out observation contexts.
- Three training seeds and one machine.
- Shared evaluation cases, not independent datasets per seed.
- Batch-one timing only.
- Different compatible rules may be selected by different methods.
- Fixed candidate order is one baseline, not every possible symbolic method.
- Training cost is additional and excluded from inference timings.

## Decision

Keep symbolic Python search as the practical reference for this task.
Do not claim a neural speedup or a general benefit of SSMs.

A future scaling experiment should specify its search size and verifier
cost before evaluation, rather than expand the problem solely to obtain
a favorable neural result.

## Artifacts

Scripts are under autonomous_lab/. Detailed ranking and timing results
remain local under autonomous_lab/runs/, excluded from Git.

The companion JSON preserves configurations, per-seed summaries, timing
measurements, and source-file hashes.
