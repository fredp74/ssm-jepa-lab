# Independent CSV verification inputs

Two independently sampled verification inputs did not provide a robust
improvement over the existing fixed checks.

The average number of incorrect answers decreased, but coverage also
decreased. The average number of accepted environments with evaluation
errors increased. Outcomes varied substantially across check seeds.

## Protocol

The benchmark contains 92 environments:

- Eight simple controls.
- Forty-eight environments using reserved two-operation sequences.
- Thirty-two environments using reserved three-operation sequences.
- Four environments that rotate columns left.

Reserved sequences were excluded from the specified earlier development
manifests. Absence from the entire historical experiment record was not
established.

Discovery uses the fixed sequence 463, 399, 0. Learning observations,
candidate programs and resource ceilings are shared across all arms.

The reference uses the two v3 verification inputs. Ten alternative arms
each use two independently sampled inputs, selected without tool feedback
or surviving-hypothesis information. Their value pool is informed by the
known primitives.

Hypotheses are frozen before verification. Verification responses do not
filter the hypotheses. They determine whether the frozen memory is
provisionally accepted, revoked or left ambiguous.

All choices are saved before evaluation on 200 prepared tables per
environment. The ten check seeds share the same environments and tables;
they are not ten independent behavior benchmarks.

## Overall results

| Metric | Fixed verification | Independent checks, mean |
|---|---:|---:|
| Answer coverage | 67.69% | 63.25% |
| Answers provided | 12,455 | 11,638.6 |
| Correct answers | 12,267 | 11,497.8 |
| Incorrect answers | 188 | 140.8 |
| Accepted environments with errors | 11 | 12.6 |

Each arm has 18,400 evaluation questions.

Independent-check errors ranged from 12 to
196. Coverage ranged from
44.01% to
72.04%.

A lower absolute error count cannot by itself establish better
verification when the procedure also answers fewer questions.

## All check seeds

| Arm | Coverage | Incorrect answers | Accepted environments with errors |
|---|---:|---:|---:|
| independent_126101 | 67.69% | 188 | 11 |
| independent_126102 | 53.55% | 76 | 12 |
| independent_126103 | 72.04% | 196 | 15 |
| independent_126104 | 72.04% | 196 | 15 |
| independent_126105 | 72.04% | 196 | 15 |
| independent_126106 | 44.01% | 12 | 4 |
| independent_126107 | 53.55% | 76 | 12 |
| independent_126108 | 53.55% | 76 | 12 |
| independent_126109 | 72.04% | 196 | 15 |
| independent_126110 | 72.04% | 196 | 15 |

No seed is selected as a winner after inspecting these results.

## Results by behavior group

All observed errors were in the three-operation group.

The fixed reference answered 92.88% of questions in the reserved-pair
group, with zero observed errors. Independent checks had lower average
coverage in that group, also with zero observed errors.

The rotation operation was rejected in all four configurations by every
arm. This does not establish detection of arbitrary unknown behavior.

Independent checks rejected some erroneous fixed-reference acceptances,
but could also accept environments on which the reference abstained.
Counting only corrected reference failures would miss those new
acceptances.

The reported fixed_correct_answers_lost_by_environment metric sums
positive per-environment decreases in correct-answer counts. It is not
a question-level count of matching predictions lost.

## Resources and persistence

Every arm used 524 calls and
1858 cells. Actual input-byte consumption can differ.

Learning observations were computed once per environment and shared
between verification arms. These are simulated logical tool-call
comparisons, not repeated deployments against an external service.

Shared generated-probe response precomputation took
35.00 seconds. Shared learning took
0.73 seconds on the reported machine.
These timings are descriptive, not replicated benchmarks.

Payload and prediction roundtrip checks passed for
1012/1012 memories. Evaluation used
candidate identities validated against the reloaded memory payloads.

## Interpretation and limitations

Independent sampling alone does not make two checks sufficiently
informative. Some draws reduce errors through increased abstention;
others allow more erroneous acceptances.

Candidate agreement remains conditional on the candidate language.
A true behavior outside that language can agree with retained programs
on observed inputs and disagree later.

Three operations do not automatically imply an unrepresentable behavior.
The evaluator separately records whether any expanded-family program
matches the full evaluation response vector. That finite-table
diagnostic is not a proof of global equivalence.

The environments and value pools are synthetic. The same evaluation
tables are shared across related environments, so question counts are
not independent replications.

There is no neural training and no demonstrated SSM/JEPA advantage.
This experiment does not establish robust autonomous learning or a
general verification guarantee.

These results are now consulted development evidence for future changes.

## Reproduction

After the generated-observation experiments:

1. python autonomous_lab/prepare_csv_independent_checks.py
2. python autonomous_lab/evaluate_csv_independent_checks.py
3. python autonomous_lab/summarize_csv_independent_checks.py

Preparation records dependency and input hashes. Evaluation checks them
before running. Existing destinations are not overwritten.

Raw runs remain local under autonomous_lab/runs/csv_independent_checks_01.
