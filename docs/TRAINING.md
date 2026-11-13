# Training and generation

## Install

Python 3.10+ and PyTorch 2.6+ are required.

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v

For CPU-only PyTorch, follow the official installation instructions for your
platform. The SSM implementation uses reference PyTorch operations, not
optimized official Mamba kernels.

## Prepare English text

    python prepare_english.py

This prepares a bounded TinyStories subset using separate training and
validation source files. Defaults read at most 16 MiB and 1 MiB respectively.
Preparation keeps complete stories, normalizes Unicode, removes exact
duplicates and exact split overlap, and records source revision and hashes.

Dataset: https://huggingface.co/datasets/roneneldan/TinyStories

Data license: CDLA-Sharing-1.0. The corpus is synthetic English narrative,
not financial training data. The data stays local under data/english.

## Start a new model

    python ssm_jepa.py init --from-scratch --data data/english/corpus.json --run runs/english.pt --width 256 --layers 8 --state-size 8 --ffn-mult 2 --lr 0.0003 --warmup 200 --decay-steps 20000 --variance-weight 0.01
    python ssm_jepa.py train --run runs/english.pt --steps 1000

This configuration has 4,465,792 language-model parameters. Teacher and
predictor parameters are additional training-time components. The model uses
256 byte values, 256-byte contexts and future 32-byte representation blocks.
The latent objective is JEPA-inspired, not a canonical JEPA reproduction.

A 1,000-step run starts training; it is not a convergence guarantee.
Initialization refuses to overwrite an existing checkpoint.

## Continue an existing run

    python ssm_jepa.py train --run runs/english.pt --steps 4000

Do not rerun init or corpus preparation for an existing run. The checkpoint
stores the model, optimizer, EMA teacher, predictor, sampling state, learning-rate
settings, replay memory and training history. Warmup and cosine decay follow
the global saved step, with a decay floor of 10% of the initial learning rate.

Saves occur every 100 steps and at completion. Ctrl+C requests a stop after
the current step and saves. Forced termination may lose work since the last save.

## Evaluate and generate

    python ssm_jepa.py eval --run runs/english.pt --samples 64
    python ssm_jepa.py generate --run runs/english.pt --prompt "Once upon a time, " --bytes 500 --temperature 0.7 --seed 42

Generation continues the prompt. The present model is not instruction-tuned.
Temperature zero selects greedy generation.

Evaluation reports bits per byte on reserved windows. Compare runs only with
matching corpus, partition, context and sampling settings. This metric does
not establish financial competence.

## Learn from new text

    python ssm_jepa.py train --run runs/english.pt --data data/new_text.txt --steps 500

Raw text, one-dimensional byte tensors in .pt files, and explicit corpus.json
manifests are supported. Manifests verify training and validation file hashes.
Without a manifest, large corpora reserve their final 500,000 bytes; smaller
corpora reserve their final fifth. Use at least 1,285 bytes with default context.

Each batch combines four incoming windows and two replay windows. Half the
256-window replay memory stays anchored to the initial corpus; the remainder
is a reservoir of incoming sampled windows. This mitigates forgetting without
guaranteeing retention. New-text training updates weights; generated answers
are not automatically used as factual training data.

## Hardware and compatibility

CPU is the tested path. On a CUDA-equipped machine, place --device cuda before
the command, for example:

    python ssm_jepa.py --device cuda train --run runs/english.pt --steps 1000

New models can configure width, layers, state size, feed-forward multiplier
and context length. Existing checkpoints retain their architecture. The
original 1.8M model remains compatible.

To initialize from the original compatible pretrained weights:

    python ssm_jepa.py init --checkpoint lm_SSM-n4_A.pt --data dataA.pt --run runs/model.pt --lr 0.00001

Checkpoints and corpora are not distributed with the repository.
