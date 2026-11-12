# SSM-JEPA Lab

Experimental sequence-model research in Python and PyTorch:
SSMs, GRUs, Transformers, JEPA objectives, continual learning,
and instruction following.

## Why

Study how small sequence models learn new tasks while retaining
previous capabilities. Experiments cover replay, forgetting,
copying, field extraction, and code recall across conversation turns.

## Current assistant

The current reference checkpoint is v09, using a GRU with causal
sliding-window attention.

The assistant is trained on constrained synthetic tasks with French
instructions. It is not a general-purpose conversational model.

- Copy a character sequence.
- Extract a named field.
- Recall a code from conversation history.
- Replace a previously supplied code.
- Answer INCONNU when no code is available.

Chat uses conversation history without updating model weights.
The v10 experiment adds training focused on longer copies.

## Repository layout

- Root Python scripts: model experiments, benchmarks, and diagnostics.
- build/assistant_v*.py: versioned assistant implementations and trainers.
- build/instructions_*/: synthetic datasets and evaluation cases.
- CSV files: experiment results.
- HANDOFF.md: development notes.

Despite its name, build/ contains Python source files.

## Requirements

Python and PyTorch are required for the assistant.
Other research scripts may have additional dependencies.

The development setup used Windows PowerShell and Python 3.13.
A complete dependency specification has not yet been established.

## Local chat

A trained checkpoint must already be available locally.
From the repository root:

```powershell
Set-Location .\build
python .\assistant_v04.py chat --run .\runs\assistant_update_v09.pt --temperature 0 --bytes 96
```

Example conversation:

```text
Vous > Retiens ce code : ab3. Reponds seulement OK.
Assistant > OK
Vous > Remplace ce code par x7q. Reponds seulement OK.
Assistant > OK
Vous > Quel code viens-je de donner ? Reponds seulement avec ce code.
Assistant > x7q
```

Commands:
- /reset clears conversation history.
- /quit exits the chat.

## Evaluation

From build/, with the checkpoint and dataset available:

```powershell
python .\assistant_v09.py eval --run .\runs\assistant_update_v09.pt --data .\instructions_v09\test.jsonl --generate --limit-per-task 16 --history-controls
```

Evaluate language-model retention separately:

```powershell
python .\assistant_v04.py eval-lm --run .\runs\assistant_update_v09.pt --samples 200
```

### Observed results

On the 292-case v09 validation set:

| Task | v09 | v10 |
|---|---:|---:|
| Copy, 2-6 characters | 80/80 | 76/80 |
| Copy, 7-8 characters | 26/32 | 29/32 |
| Field extraction | 64/64 | 63/64 |
| Code recall | 48/48 | 47/48 |
| Code replacement | 48/48 | 48/48 |
| Missing-code response | 20/20 | 20/20 |
| Total | 286/292 | 283/292 |

A successful case requires exact answers and the end marker
at every assistant turn.

v09 remains the reference checkpoint. v10 improves longer copies
but loses accuracy on other tasks.

These results describe a small synthetic validation set.
That set informed the v10 training changes, so it is not an
independent final benchmark.

A separate 228-case validation set was generated for comparison;
its results are not included here yet.

## Known limitations

- Evaluation covers a limited set of instruction templates.
- Longer copies remain less reliable.
- Training new tasks can degrade previous capabilities.
- Recall is bounded by the conversation context.
- Codes of two characters were omitted from the new validation
  because too few unused combinations remained.
- Checkpoints, downloaded corpora, and runtime logs are excluded
  from Git. Cloning alone does not provide a runnable trained model.
- The repository is an experimental workspace; a complete
  end-to-end reproduction guide is still being developed.

---

A research project by [Algotradingresearch.com](https://algotradingresearch.com).

## Research results

See [V13 experiment: repeated-character training](docs/v13-results.md)
for the three-seed comparison, final evaluation results, and limitations.
