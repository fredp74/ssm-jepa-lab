# Frozen CSV transfer evaluation

The frozen v3 procedure transferred poorly beyond its six development
compositions. The expanded candidate language had low answer coverage,
abstained on all simple controls, and made errors on an accepted
three-operation environment.

This negative result limits the interpretation of earlier development
successes.

## Protocol

The experiment contains 84 environments:

- Eight simple controls.
- Forty-eight environments using 12 ordered pairs absent from the six
  composition-development cases.
- Twenty-four environments using six three-operation sequences.
- Four environments that reverse columns, an operation absent from the
  candidate language.

The pair exclusion was checked against the six development cases, not
the entire historical experiment record.

The v3 policy, relevant dependencies, environment definitions and 200
evaluation tables were fingerprinted before execution. Evaluation
feedback was not used for test selection.

Both candidate languages were tested: at most one operation, and at
most two ordered operations. The elementary transformations and search
language remain human-defined.

The total ceiling per environment was eight calls, 24 cells and 512
input-text UTF-8 bytes. The procedure reserved separate learning and
verification inputs.

## Results

| Candidate language | Group | Answers provided | Errors | Coverage |
|---|---|---:|---:|---:|
| single | Pairs outside the six development cases | 4,600/9,600 | 998 | 47.92% |
| single | Simple controls | 800/1,600 | 0 | 50.00% |
| single | Three-operation sequences | 600/4,800 | 42 | 12.50% |
| single | Column reversal | 0/800 | 0 | 0.00% |
| composed | Pairs outside the six development cases | 600/9,600 | 0 | 6.25% |
| composed | Simple controls | 0/1,600 | 0 | 0.00% |
| composed | Three-operation sequences | 200/4,800 | 14 | 4.17% |
| composed | Column reversal | 0/800 | 0 | 0.00% |

Only provisionally accepted memories could answer. Within those
memories, every retained program had to agree on each prediction.
All other cases resulted in abstention.

The composed arm accepted only three of the 48 pair environments.
It answered none of the simple-control questions.

One three-operation environment passed verification in the composed
arm but produced 14 errors among 200 answers. Its behavior applied
rstrip, leading-zero normalization and empty-column removal.

Column reversal was detected as incompatible in all four configurations
by both candidate languages.

The single-operation arm provided more answers but made 1,040 errors
among 6,000 answers across the full benchmark.

## Behavioral distinguishability

After the agent's choices were frozen, the evaluator compared each
environment's complete response vector on the 200 evaluation tables
with every expanded-language candidate.

No expanded candidate matched all evaluation responses for any of the
24 three-operation environments or the four column-reversal
environments.

This is a finite-table witness against matching by a single candidate.
It does not prove that every individual table distinguishes the true
behavior from every candidate. Conversely, matching on the tables
would not prove global equivalence.

These response-vector comparisons were used only for diagnosis, not
to choose actions or update the agent.

## Persistence

Payload and prediction roundtrip checks passed for
168/168 memories, using catalogue and
verification inputs.

Evaluation predictions were produced from reloaded memories.
Persistence therefore passed the implemented checks while behavioral
identification and verification remained inadequate.

## Interpretation

The procedure is too conservative in some in-family environments:
it abstains even on the simple controls.

It is also insufficiently protective outside the candidate language:
an incomplete explanation can pass the fixed checks and produce
incorrect answers.

Expanding the candidate repertoire increases representational capacity
but also leaves more explanations unresolved under the available
observations. A smaller repertoire can appear more decisive while
being wrong.

Earlier zero-error development results did not establish general
reliability. The frozen transfer test exposes that limitation.

These experiments demonstrate neither robust autonomous learning nor
an SSM/JEPA advantage. They use symbolic candidate enumeration and
explicit memory, with no neural training.

## Limitations

The benchmark uses simulated tools and predefined operations.
Only four CSV configurations are used for each behavior.

The same 200 tables are shared across environments, so the 16,800
questions per arm are not independent replications.

Three-operation and column-reversal groups probe selected forms of
candidate-family mismatch, not arbitrary unknown tools.

Both arms share resource ceilings but consume different actual
resources. The comparison does not establish equal-cost superiority.

The v3 procedure and its checks were informed by previous development
experiments. This evaluation was frozen before execution, but its
results must now be treated as consulted development evidence for
subsequent method changes.

## Reproduction

After the composition v3 experiments:

1. python autonomous_lab/prepare_csv_transfer.py
2. python autonomous_lab/make_csv_transfer_evaluator.py
3. python autonomous_lab/evaluate_csv_transfer.py
4. python autonomous_lab/summarize_csv_transfer.py

Preparation records fingerprints. The evaluator checks them and
refuses changed dependencies or prepared inputs.

Generators and experiments refuse to overwrite existing destinations.
Raw results remain local under autonomous_lab/runs/csv_transfer_01.

The next research question is how to allocate observations more
effectively and assess candidate-family mismatch, rather than adding
checks tailored repeatedly to each observed failure.
