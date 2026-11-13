# Contributing to ssm-jepa-lab

Contributions, replications, critiques and negative results are welcome.

This repository is an experimental research workspace, not a finished
framework or a self-learning language model. It contains several research
directions, including SSM experiments, latent objectives, explicit memory
and symbolic learning through tool interaction.

## Start here

Read these reports for the latest CSV experiments:

- [Executable skill acquisition](docs/csv-executable-skill-acquisition.md)
- [Consensus predictions and exploration controls](docs/csv-consensus-and-exploration.md)

The second report includes an important negative result: an optimized
fixed pair of tests outperformed our current adaptive heuristic.

Historical experiments have different dependencies and local artifacts.
There is not yet a single command that reproduces the entire repository.
If setup instructions are missing for an experiment, please open an issue.

## Accessible first contributions

### 1. Reproduce a reported result

Choose one documented experiment and record:

- The commit and commands used.
- Python version, dependencies and operating system.
- Seeds and relevant input artifacts.
- Your results and any differences from the report.
- Runtime and resource consumption where practical.

Reporting an obstacle to reproduction is useful too.

### 2. Add a focused counterexample

Construct a small input that exposes an incorrect prediction,
an inadequate verification check or an unsupported assumption.

Explain the expected behavior and why the example matters.
Keep discovery inputs, verification inputs and evaluation inputs separate.

### 3. Improve one baseline

Compare a proposed method with a simple reference under explicit budgets.

Report both prediction quality and cost: coverage, errors, abstentions,
tool calls, cells, input bytes and offline computation where relevant.

A method that does not outperform the reference is still worth reporting.

## Other research directions

- Experiment selection that optimizes useful predictions.
- Detection of behavior outside the candidate language.
- Composition and synthesis of executable programs.
- Memory persistence and recovery after behavior changes.
- Learned representations with a measurable practical benefit.
- Clearer setup instructions and smaller reproduction examples.

These are research questions, not claims that the current system has
solved them.

## Proposing an experiment

Open an issue describing:

1. The question or failure mode.
2. The proposed method and reference baseline.
3. What information each method may access.
4. Resource budgets and evaluation metrics.
5. How development and evaluation data will be separated.

For substantial experiments, discuss the protocol before running the
final evaluation. Do not tune on evaluation results and then describe
that same evaluation as independent.

Small fixes and focused counterexamples can be submitted directly.

## Submitting a pull request

- Keep the change focused.
- Explain the problem, change and validation.
- Distinguish observations from interpretations.
- Include seeds and reproduction commands when relevant.
- Preserve raw results rather than silently rewriting them.
- Document limitations and unsuccessful outcomes.
- Avoid committing checkpoints, large generated datasets or local runs.
- Use English for shared documentation.

Tests should check meaningful behavior. In particular, persistence checks
should compare predictions before saving and after reloading, rather
than merely reading the same file twice.

## Interpreting results

Please distinguish:

- Learning model parameters from updating memory or hypotheses.
- Selecting predefined programs from inventing new primitives.
- New input tables from genuinely unseen behavior families.
- Accuracy among answered queries from overall coverage.
- Passing a few checks from a general correctness guarantee.

Zero observed errors within a candidate family does not establish
reliability outside that family.

## License

Check the repository's current license before reusing or redistributing
code. This contributing guide does not grant additional license rights.
