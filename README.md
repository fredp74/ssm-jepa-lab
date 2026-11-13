# SSM-JEPA Language Model

A small, runnable language model built with selective state-space layers
and a JEPA-inspired latent prediction objective.

The project focuses on one implementation: training, checkpoint recovery,
recurrent text generation and continued learning from new text.

## Status

The current model has approximately 1.8 million language-model parameters,
plus a training-time latent predictor and EMA teacher.

It generates text, but the current locally trained checkpoint does not
yet produce consistently coherent language. This is an experimental
foundation for a larger language model, not a finished conversational
assistant.

## Architecture

- Byte vocabulary: 256 values, without an external tokenizer.
- Width: 256.
- Six selective SSM blocks with state dimension four.
- Next-byte language-model head.
- Predictor for representations of future 32-byte blocks.
- Teacher updated by exponential moving average.
- Combined loss: next-byte cross-entropy plus 0.1 times latent loss.

The latent objective is inspired by JEPA; this is not a reproduction of
a canonical JEPA architecture.

Training uses parallel sequence processing. Generation carries the SSM
states forward one byte at a time.

## Requirements

Python 3.10 or newer and PyTorch 2.6 or newer.

Development has been exercised on CPU, including Windows PowerShell.

Install dependencies:

    python -m pip install -r requirements.txt

For a CPU-only PyTorch installation, use the appropriate official PyTorch
installation command for your platform.

## Train from scratch

Provide a UTF-8 plain-text corpus, for example data/train.txt.
The file must be large enough for training and a reserved evaluation
partition; at least 1,285 bytes are needed with the current context length.
Useful language learning requires substantially more data and training.

    python ssm_jepa.py init --from-scratch --data data/train.txt --run runs/model.pt --lr 0.0001
    python ssm_jepa.py train --run runs/model.pt --steps 500
    python ssm_jepa.py eval --run runs/model.pt

The learning rate is configurable at initialization. It is stored with
the optimizer in the checkpoint.

## Initialize from compatible pretrained weights

For users who already have the original SSM language-model weights:

    python ssm_jepa.py init --checkpoint lm_SSM-n4_A.pt --data dataA.pt --run runs/model.pt --lr 0.00001

Initialization requires an exact architecture match. Existing destination
checkpoints are not overwritten.

Weights and corpora are not distributed in this repository.

## Generate text

    python ssm_jepa.py generate --run runs/model.pt --prompt "The man looked at " --bytes 500 --temperature 0.7

Use temperature 0 for greedy generation. Generation continues the prompt;
the model has not been trained as an instruction-following assistant.

## Resume training

    python ssm_jepa.py train --run runs/model.pt --steps 500

The checkpoint contains model weights, teacher, predictor, optimizer,
sampling-generator state, replay memory and training history.

Training saves every 100 steps and at the end. Ctrl+C requests a stop
after the current step and saves. A crash or forced process termination
can lose work since the last completed save.

## Learn from new text

    python ssm_jepa.py train --run runs/model.pt --data data/new_text.txt --steps 500

This updates model parameters using self-supervised prediction.
It does not automatically train on generated answers.

Each batch combines four incoming windows and two replay windows.
Half of the 256-window memory remains anchored to the initial corpus;
the other half is updated as a reservoir of sampled incoming windows.

Replay mitigates forgetting; it does not guarantee retention.

## Data and evaluation

Input can be a raw text file or a .pt file containing a one-dimensional
integer tensor with values in [0,255].

For corpora of at least one million bytes, the final 500,000 bytes are
reserved for evaluation. Smaller corpora reserve the final fifth.

The context length is 256 bytes. Evaluation reports bits per byte
on fixed windows in the reserved partition; lower is better.

    python ssm_jepa.py eval --run runs/model.pt --samples 256
    python ssm_jepa.py eval --run runs/model.pt --data data/another_corpus.txt --samples 256

Different corpora or evaluation settings must not be compared as if they
were the same benchmark. Evaluation loss alone does not establish coherent
generation or comprehension.

## Tests

    python -m unittest discover -s tests -v

Tests cover recurrent/parallel agreement, causality, exact continuation,
protected replay memory, new-data training, checkpoint persistence,
interruption handling and initialization without pretrained weights.

## Files

- ssm_jepa.py: model, training, evaluation and generation.
- ssm_par.py: selective SSM layers and parallel scan.
- tests/test_ssm_jepa.py: functional tests.
- requirements.txt: dependencies.

Local datasets and checkpoints are excluded from Git.
Earlier experiments remain available in Git history.

## Contributing

See CONTRIBUTING.md. Contributions should directly improve the language
model, its training system, data pipeline or evaluation.
