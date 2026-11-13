# CSV consensus predictions and exploration controls

The optimized fixed pair outperformed the current adaptive heuristic.
An advantage of adaptation has not been established.

## Scope

These experiments revisit the 26 environments left ambiguous by the
executable-skill acquisition pilot. Candidate programs combine predefined
individual transformations with CSV configurations.

A question is answered only when every surviving program produces the
same output. Otherwise, the system abstains. All program aliases are
checked individually.

This is symbolic inference within a programmed hypothesis language.
No neural training is performed. These results do not demonstrate
open-ended skill invention or an SSM/JEPA learning advantage.

## Retrospective diagnostics

On the original evaluation questions, consensus answered
658/2,600
correctly, with 25.3% coverage.

One additional targeted call per environment increased coverage to
38.4%, yielding
998 correct answers and zero observed errors.
All environments selected test 11.

Both diagnostics reused the original evaluation tables. Choosing the
same test everywhere did not establish a benefit from adaptation.

## Initial exploration comparison

Each strategy received up to two additional calls per environment.
The total resource ceiling was eight calls, 24 cells and 512 input-text
UTF-8 bytes, including the shared initial observations.

A new set of 200 tables was evaluated in each of the 26 environments.

| Strategy | Coverage |
|---|---:|
| Adaptive heuristic | 65.7% |
| Simple fixed sequence | 36.3% |
| Random, mean across 20 seeds | 33.6% |

Random coverage ranged from 27.4% to 40.3%.
Each strategy used 52 additional calls per run. There were zero observed
errors among answered queries.

The fixed sequence started with test 11, informed by the previous
diagnostic, and then followed catalogue order. It was not optimized.

The adaptive heuristic minimizes the expected number of remaining
catalogue-response classes under a uniform class prior.

## Optimized fixed-pair control

All 1,892 ordered pairs
were searched using the original 100 development tables and reused
environments. Selection maximized consensus coverage, with resource cost
and deterministic order breaking ties.

The chosen pair, **10 then 31**, was saved before evaluation
on another 200 new tables per environment. Adaptive decisions were
reused unchanged.

| Strategy | Correct answers | Coverage | Additional calls |
|---|---:|---:|---:|
| Optimized fixed pair | 3,889/5,200 | 74.8% | 52 |
| Adaptive heuristic | 3,535/5,200 | 68.0% | 52 |

The fixed pair gained 354
answers, or 6.8
percentage points. Both strategies had zero observed errors.

The fixed pair had higher coverage in 15 environments;
the adaptive heuristic had higher coverage in 7;
4 were tied.

Total cell consumption was 520 for the fixed pair
versus 516 for the adaptive heuristic.
Total input-text consumption was 3,562
versus 3,424 bytes.

Call counts were equal and resource ceilings were shared, but actual
cell and byte consumption differed.

## Interpretation and limitations

The optimized fixed pair is the stronger reference in this experiment.

Reducing candidate-class uncertainty and maximizing prediction coverage
are different objectives. The fixed-pair search optimized development
coverage directly; the adaptive heuristic optimized class uncertainty.
This comparison does not rule out better adaptive policies.

The fixed pair was optimized offline using simulated development
environments and their known programs. That search cost is not included
in the reported tool-call budget.

The environments and transformation language were reused. New tables
measure transfer to new inputs within that family, not discovery of
unseen transformation families or arbitrary tools.

The true simulated program remains among the candidates. Under that
assumption, unanimous predictions must agree with it. Zero observed
errors do not establish reliability outside the candidate family.

The 5,200 questions reuse 200 tables across 26 related environments.
They are not 5,200 independent experimental replications.

## Reproduction

Run the following scripts from the repository root, after the
executable-skill acquisition pilot:

1. python autonomous_lab/diagnose_csv_ambiguity.py
2. python autonomous_lab/evaluate_csv_targeted_probe.py
3. python autonomous_lab/compare_csv_exploration.py
4. python autonomous_lab/compare_csv_optimized_fixed.py
5. python autonomous_lab/summarize_csv_consensus.py

Experiment scripts refuse to overwrite existing output directories.
Raw results remain local under autonomous_lab/runs.
The companion JSON records source summaries and SHA-256 hashes.
