# Local hypothesis repair

## Scope

A symbolic agent revises a provisional hypothesis after observing a
counterexample. This experiment uses no neural training, SSM, or JEPA.

The programmed repair family contains 4,521 functions: a member of the
original 137-rule family, optionally with one output flipped.

## Protocol

Only the 215 provisionally accepted functions from the preceding
budgeted acquisition experiment are included.

- Preserve every original observation.
- Monitor up to eight previously unqueried inputs.
- Revoke the original hypothesis on the first counterexample.
- Search the repair family for hypotheses consistent with all evidence.
- Ask distinguishing questions while reserving budget for verification.
- Freeze a unique repair before up to four verification queries.
- Consult at most 24 of the 32 inputs in total.
- Evaluate accepted hypotheses on at least eight unqueried inputs.

A surviving hypothesis remains provisional.

## Results

| Group | Functions | Outcome |
|---|---:|---|
| Known rules | 137 | All remain correct |
| Near misses | 71 | 29 exact repairs; 42 incorrect hypotheses remain unchanged |
| Random outside family | 7 | Six repairs revoked; one incorrect repair survives |

The 29 accepted near-miss repairs predict all 373 evaluator-only outputs
correctly. The 42 unchanged near-miss hypotheses retain 42 errors across
545 evaluator-only outputs.

The accepted random outside-family repair predicts only six of its
15 evaluator-only outputs correctly, despite surviving four checks.

## Interpretation

Repair succeeds when a counterexample is observed and the repair family
contains the appropriate correction. Failure to encounter the exceptional
input remains a major obstacle.

The near-miss test family was constructed to differ from a known rule
on one input. The repair representation directly matches that structure.
This is a targeted mechanism check, not general rule invention.

Preserving evidence prevents a repair from contradicting known facts.
It does not guarantee correctness on unknown inputs.

## Limitations

- One programmed repair representation.
- One monitoring path per function.
- Selected cases: only hypotheses accepted by the preceding experiment.
- Deterministic, noiseless environments.
- No matched no-repair control with identical new queries.
- High output accuracy does not establish exact rule identification.
- Four successful verification queries do not certify a repair.

## Decision

Preserve this milestone without promoting the repair method to a general
learning mechanism.

The tested lifecycle is:
provisional hypothesis -> monitoring -> revocation -> repair -> new checks.

Future work must measure the cost and false acceptance risk of broader
repair representations, without discarding counterexamples.

## Artifacts

The implementation is autonomous_lab/local_repair.py.
Detailed local results remain under autonomous_lab/runs/local_repair_01/,
excluded from Git. The companion JSON stores configuration, aggregate
results, and a hash of the detailed source file.
