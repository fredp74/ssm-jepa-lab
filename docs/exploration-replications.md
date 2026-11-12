# Sequential exploration: three-ensemble replication

## Question

Can a frozen neural ensemble choose more useful questions than random
sampling in the three-bit affine-rule environment?

## Protocol

Three independently trained ensembles contain three diagonal SSMs each.
Training includes zero to four observations per episode.

Each evaluation episode allows four distinct questions. Final answers
use the mean probabilities of the same ensemble for every policy.

Policies:
- Random sampling without replacement.
- Maximum predictive entropy of the ensemble mean.
- Ensemble disagreement: entropy of the mean minus mean member entropy.

No model parameters are updated during evaluation.

The random reference exhaustively covers 70 question sets, 24 orders,
and all 16 rules. Guided policies use 24 repetitions per rule, including
random tie-breaking where applicable.

## Results

| Ensemble | Exhaustive random | Entropy | Disagreement |
|---|---:|---:|---:|
| 01 | 59.06% | 62.50% | 60.94% |
| 02 | 58.34% | 62.24% | 54.69% |
| 03 | 58.36% | 56.25% | 62.50% |

Mean difference from each ensemble's exhaustive random reference:
- Entropy: +1.74 percentage points.
- Disagreement: +0.79 percentage points.

Neither guided policy improves on random for every ensemble.

## Interpretation

The first ensemble suggested a benefit, but replication showed substantial
dependence on the trained ensemble. A positive average across three
ensembles does not establish a reliable improvement.

Predictive uncertainty and ensemble disagreement are not guaranteed
measures of information gain about the hidden rule.

## Limitations

- One small, familiar family of 16 rules.
- Only three ensembles.
- Shared evaluation cases; counts are not independent trials.
- Guided-policy repetitions can repeat identical trajectories.
- Random performance is exhaustive; guided-policy tie-breaking is sampled.
- Evaluation includes both training and held-out four-question contexts.
- The benchmark has been consulted repeatedly during development.
- Prior neural training is required; this is not learning from nothing.

## Decision

Close this experiment without selecting a guided policy as a demonstrated
improvement. Preserve the protocol and results before changing the design.

The system supports interaction-driven state adaptation, but has not
demonstrated robust autonomous acquisition of new rule families.

## Artifacts

Scripts are in autonomous_lab/. Detailed local results and checkpoints
are under autonomous_lab/runs/ and remain excluded from Git.
