# CSV composition acquisition: development pilot

Expanding the candidate language from at most one operation to at most
two ordered operations increased answer coverage from
16.7% to 33.0% in this development
pilot. The expanded repertoire used more tool calls.

## Experimental scope

The benchmark contains 24 simulated environments: six compositions
across four CSV configurations.

The elementary operations are implemented by humans. Candidate generation
exhaustively enumerates zero, one or two distinct operations in order,
combined with CSV configurations. The expanded language contains 6,960
candidate programs.

This experiment measures selection and retention of composed programs
within that language. It does not demonstrate invention of new primitive
operations, open-ended program synthesis or autonomous language learning.

Some compositions were explored in earlier experiments. This is a
development benchmark, not an evaluation on demonstrably unseen
behavior families.

## Procedure

Both arms start without observations of the current environment.

Discovery selects catalogue probes to reduce candidate uncertainty.
The total budget is eight calls, 24 cells and 512 input-text UTF-8 bytes
per environment. Discovery reserves resources for two verification calls.

Candidate programs are frozen before verification. A provisional status
requires unanimous predictions matching both verification responses.
Disagreement between candidates causes abstention; a response outside
their predictions revokes the proposal.

During evaluation, only provisional memories may answer. Every retained
program must agree on the answer to the current input.

A set of 200 evaluation tables is shared across the 24 environments.
Evaluation feedback is not used to select tests or candidate programs.

## Results

| Metric | At most one operation | At most two operations |
|---|---:|---:|
| Environments | 24 | 24 |
| Provisional environments | 4 | 8 |
| Answers provided | 800 | 1,584 |
| Correct answers | 788 | 1,584 |
| Incorrect answers | 12 | 0 |
| Evaluation questions | 4,800 | 4,800 |
| Answer coverage | 16.7% | 33.0% |
| Tool calls | 104 | 134 |
| Cells | 473 | 563 |
| Input-text UTF-8 bytes | 2,799 | 3,323 |

The composed arm answered 1,584 questions with zero
observed errors. It remained verification-ambiguous in 16 environments.

Its accepted environments involved leading-zero normalization followed
by row sorting, and decimal-comma conversion followed by row sorting.

The single-operation arm passed verification in four environments but
made 12 evaluation errors. Passing two checks is therefore
not a general correctness guarantee.

The arms share resource ceilings, but actual consumption differs.
These results do not establish better performance at equal actual cost.

## Computation

Shared catalogue precomputation took
2.697 seconds on the
reported machine.

Search and verification took
0.140 seconds in the
single-operation arm and
1.021 seconds in the composed
arm.

These are descriptive measurements from one run, not replicated
performance benchmarks.

## Persistence limitation

Evaluation predictions use programs loaded from saved memory files.
However, the reported memory_roundtrip_passed field compares two reads
of the same file. It does not compare predictions before saving and
after reloading.

That field must not be interpreted as a valid prediction-persistence
test. A corrected experiment should compare both the in-memory payload
and predictions before and after serialization. Raw results are preserved.

## Interpretation

The expanded repertoire supports some behaviors that the initial
repertoire cannot represent adequately. Acquisition remains partial:
16 of 24 composed-arm environments are not accepted.

Zero observed errors in the composed arm must be interpreted within the
simulated candidate family, which contains the true programs. It does
not establish reliability for arbitrary external tools.

The 4,800 questions reuse 200 tables across related environments and are
not 4,800 independent replications.

There is no neural training in this experiment and no demonstrated
SSM/JEPA advantage.

## Reproduction

From the repository root:

1. python autonomous_lab/prepare_csv_composition.py
2. python autonomous_lab/acquire_csv_compositions.py
3. python autonomous_lab/summarize_csv_compositions.py

The experiment scripts refuse to overwrite existing output directories.
Raw results remain local under autonomous_lab/runs/csv_composition_01.
The companion JSON records the source hash, protocol and summary.
