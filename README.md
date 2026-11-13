# SSM-JEPA Finance

**A financial language model built to connect quantitative evidence, business value and macroeconomic risk.**

## The ambition

Build a financial research model that can examine an investment through
three complementary lenses and explain how they fit together.

The project draws inspiration from three investment traditions:

- **Jim Simons:** quantitative research, probability and empirical validation.
- **Warren Buffett:** business quality, intrinsic value and long-term capital allocation.
- **Ray Dalio:** economic cycles, diversification and portfolio risk.

The goal is to combine these perspectives in one model: understand the
business, examine the data, and place both in their economic and risk context.

These references express the research direction. They do not imply an
association with these investors or replication of their proprietary methods.

## What the model should deliver

Given financial documents, market evidence and an analysis date, the intended
system should produce a research view that answers:

- What does the quantitative evidence support, and how reliable is it?
- What drives the business, its cash generation and its valuation?
- Which macroeconomic conditions and portfolio exposures matter?
- Where do these perspectives agree or conflict?
- What assumptions, missing information or risks could change the conclusion?

Sources, dates, calculations and uncertainty should remain visible in the
answer. Financial conclusions must be evaluated against evidence available
at the stated date.

## Why SSM and JEPA?

**Selective state-space models** provide the sequence-processing backbone,
with recurrent inference and a persistent internal state during generation.

**JEPA-inspired latent prediction** adds an objective for learning future
representations alongside ordinary language prediction.

The research question is whether this combination can support useful,
efficient financial language modeling and continued learning. Its value
must be demonstrated through language quality and financial accuracy.

## Where the project stands

An English language-model prototype is running, with a configurable SSM
backbone, feed-forward sublayers, an EMA teacher and recurrent text generation.
The current English configuration has approximately **4.47 million parameters**
in the language model.

Training currently uses TinyStories to establish basic English generation.
Financial specialization and the integrated research workflow are the next
stages. The present model still makes language errors; financial competence
and investment performance have not been established.

## Development priorities

1. Build coherent, reliable English generation.
2. Specialize on sourced, dated financial documents.
3. Support verifiable quantitative and fundamental calculations.
4. Integrate macroeconomic scenarios and portfolio risk.
5. Evaluate combined financial research and retention after model updates.

See the [roadmap](docs/ROADMAP.md).

## Run and contribute

The implementation supports CPU training, resumable checkpoints, recurrent
generation and continued training on new text. CUDA execution is available
in the reference code; functional tests have been run on CPU.

- [Installation, training and generation](docs/TRAINING.md)
- [Contribution guide](CONTRIBUTING.md)
- [Development roadmap](docs/ROADMAP.md)

Weights, corpora and local training outputs stay outside Git.
