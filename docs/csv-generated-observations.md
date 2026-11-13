# Generated CSV observations: fixed versus adaptive selection

A bank of automatically generated inputs substantially improved
coverage within the predefined candidate family. A fixed sequence
selected from the simple-control experiment matched adaptive selection
on all 84 environments, with much less online search computation.

Errors on three-operation environments remained unresolved.

## Generated input bank

The bank contains 512 inputs constructed from generic templates and
bounded random sampling. Values are informed by the known primitive
operations. Hidden environment programs are not used to generate inputs.

This is selection from a generated finite bank, not unrestricted input
synthesis or invention of new primitive operations.

Both arms use the same expanded candidate family: up to two ordered
operations combined with CSV configurations.

Selection minimizes expected remaining response classes under a uniform
prior over original-catalogue response classes. Learning inputs and
verification inputs remain unchanged from v3.

## Simple-control comparison

Eight environments cover identity and casefold across four CSV
configurations.

| Observation bank | Correct answers | Coverage | Calls | Cells |
|---|---:|---:|---:|---:|
| Original catalogue | 0/1,600 | 0.0% | 40 | 192 |
| Generated inputs | 1,306/1,600 | 81.6% | 56 | 192 |

Both arms had zero observed errors among answered questions.
The generated bank used more calls, the same number of cells and fewer
input-text bytes.

All eight generated-bank runs selected tests 463, 399 and 0.
That observation motivated a fixed-sequence reference.

Passing verification did not imply unique program identification.
Some accepted memories retained multiple hypotheses and still abstained
on evaluation questions.

## Fixed versus adaptive comparison

The sequence **463, 399, 0** was frozen before evaluation on new tables
across the 84 previously examined environments.

These environments were reused development evidence, including the
earlier transfer failures. This is not an independent evaluation of
previously unexamined behavior families.

Both arms share the generated bank, candidate family, learning inputs,
verification inputs and resource ceilings:

- Eight total calls.
- Twenty-four cells.
- Five hundred twelve input-text UTF-8 bytes.

Discovery receives up to four calls and eight cells. The fixed arm
follows its three-test sequence, skipping infeasible tests without
substituting other tests.

| Strategy | Group | Answers | Errors | Coverage |
|---|---|---:|---:|---:|
| generated_adaptive | heldout_development_pair | 7,129/9,600 | 0 | 74.26% |
| generated_adaptive | simple_control | 1,360/1,600 | 0 | 85.00% |
| generated_adaptive | three_operations | 3,657/4,800 | 135 | 76.19% |
| generated_adaptive | unrepresented_primitive | 0/800 | 0 | 0.00% |
| generated_fixed | heldout_development_pair | 7,129/9,600 | 0 | 74.26% |
| generated_fixed | simple_control | 1,360/1,600 | 0 | 85.00% |
| generated_fixed | three_operations | 3,657/4,800 | 135 | 76.19% |
| generated_fixed | unrepresented_primitive | 0/800 | 0 | 0.00% |

Each strategy answered 12,146 of
16,800 questions, with 12,011
correct answers and 135 errors.

Coverage was equal in 84/84 paired environments.
The aggregate error counts were also equal. This does not establish
that all individual predictions or retained hypotheses were identical.

Both strategies used 524 calls.
Cell consumption was 1846 for adaptive selection
and 1848 for the fixed sequence. Input-text consumption
was 11,353 versus
11,360 bytes.

## Computation

In the simple-control experiment, generated-bank response precomputation
took 33.14 seconds,
versus 2.70 seconds
for the original catalogue.

In the 84-environment comparison, generated-bank precomputation took
34.41 seconds and was
shared by both arms.

Cumulative search and verification time was
39.07 seconds for the
adaptive arm and 0.89 seconds
for the fixed arm.

These are descriptive timings from one run, not replicated benchmarks
or complete end-to-end deployment costs.

## Persistence

Payload and prediction roundtrip checks passed for
16/
16 memories in the controls
experiment and 168/
168 in the larger comparison.

The checks compare predictions on catalogue and verification inputs
before saving and after reloading. Evaluation uses reloaded programs.

## Interpretation

The generated bank offers more useful observations under the cell
budget. No benefit from adaptive selection was measured over the
selected fixed sequence in this experiment.

The candidate family represents the tested two-operation compositions,
but not every three-operation behavior. The latter group produced
135 errors per arm despite passing checks in some environments.

Agreement among retained candidates cannot guarantee correctness when
the true behavior is outside the candidate family. Column reversal was
detected as incompatible, but that success does not establish general
unknown-behavior detection.

The expanded repertoire, generated bank and explicit memory remain
human-designed symbolic mechanisms. There is no neural training and no
demonstrated SSM/JEPA advantage.

## Limitations

The fixed sequence was selected using the simple-control results.
The larger comparison was frozen before execution, but the environments
had already been examined in earlier experiments.

Fresh evaluation tables do not make these behavior families unseen.
Tables are shared across environments, so question counts are not
independent experimental replications.

The comparison with the earlier original-catalogue transfer experiment
uses different evaluation tables. Its percentages are not a paired
question-level comparison.

Any subsequent method change must treat these results as consulted
development evidence.

## Reproduction

From the repository root, with prior composition artifacts available:

1. python autonomous_lab/prepare_csv_counterexamples.py
2. python autonomous_lab/make_csv_counterexample_evaluator.py
3. python autonomous_lab/evaluate_csv_counterexamples.py
4. python autonomous_lab/prepare_csv_generated_transfer.py
5. python autonomous_lab/make_csv_generated_transfer_evaluator.py
6. python autonomous_lab/evaluate_csv_generated_transfer.py
7. python autonomous_lab/summarize_csv_generated_observations.py

Generators and experiments refuse to overwrite existing destinations.
Prepared protocols fingerprint dependencies and input files.
Raw runs remain local under autonomous_lab/runs.
