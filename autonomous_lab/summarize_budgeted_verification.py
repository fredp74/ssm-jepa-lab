import hashlib
import json
from pathlib import Path

destination = Path("docs/budgeted-verification.md")
summary_path = Path("docs/budgeted-verification.json")
if destination.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

sources = {
    "budgeted_acquisition": Path(
        "autonomous_lab/runs/budgeted_five_bits_01/results.json"
    ),
    "continuous_verification": Path(
        "autonomous_lab/runs/continuous_verification_01/results.json"
    ),
}
summary = {}
for name, path in sources.items():
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    summary[name] = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "config": data["config"],
        "summaries": data["summaries"],
    }

text = """# Budgeted acquisition and continuous verification

## Environment

Five input bits give 32 possible inputs. The programmed hypothesis family
contains 137 functions: XOR combinations of at most two monomials of
degree at most two.

The environment is deterministic and noiseless. This is a symbolic
experiment, without SSM or JEPA components.

## Budgeted acquisition

The agent may ask up to 12 discovery questions, followed by up to four
verification questions. Its proposal is frozen before verification.

Evaluator-only answers are never returned to the agent.

| Function group | Cases | Provisional acceptances | False acceptances |
|---|---:|---:|---:|
| Known family | 137 | 137 | 0 |
| Near misses | 100 | 71 | 71 |
| Random outside family | 100 | 7 | 7 |

For known functions, the agent asks 11.15 questions on average and predicts
all remaining outputs correctly.

Near misses differ from a known function on one input. Accepted near-miss
hypotheses achieve 92.4% output accuracy, but all 71 accepted rules are
incorrect. High output accuracy is not exact rule identification.

The accepted random outside-family cases achieve 54.1% output accuracy.

## Continuous verification

Only the 215 provisionally accepted functions enter this experiment.
Each has five sampled verification paths.

Policies:
- Random usage: query uniformly from all 32 inputs, allowing repetition.
- Unseen audit: query uniformly from inputs not already checked.

A first observed counterexample immediately revokes the hypothesis.
No repair is performed in this experiment.

### Near-miss hypotheses

There are 71 functions and 355 paths.

| Additional queries | Detected by random usage | Detected by unseen audit |
|---|---:|---:|
| 4 | 57/355 | 79/355 |
| 8 | 90/355 | 147/355 |
| 16 | 159/355 | 284/355 |
| 32 | 245/355 | 355/355 |
| 64 | 316/355 | 355/355 |

Unseen audit revokes all incorrect near-miss hypotheses after 10.35
additional queries on average.

Random usage leaves 39 paths unresolved after 64 queries.
Its mean detection delay among revoked paths is 20.95 queries; this
excludes unresolved paths and is not an unconditional detection-time mean.

### Other groups

All 35 paths for the seven accepted random outside-family functions are
eventually revoked by both policies.

No correct known-family hypothesis is revoked.
Unseen audit completes verification of known functions in 20.85
additional queries on average. Random usage completes only 28 of 685
known-function paths within 64 queries.

## Interpretation

Keep separate statuses:
- Provisional: compatible with available evidence.
- Revoked: contradicted by a new observation.
- Fully verified: checked on every input in this finite domain.

Checking unobserved inputs reduces redundant interaction.
Complete detection by unseen audit is expected on a finite domain; it is
not evidence of a general solution to verification.

## Limitations

- Programmed hypothesis family and synthetic environments.
- No noise, stochastic outputs, or hidden state.
- Five paths share each function and are not independent functions.
- Near-miss and random test distributions are deliberately constructed.
- No neural learning or autonomous invention of concepts is demonstrated.

## Next question

After revocation, can the system revise its hypothesis while preserving
every observation and counterexample?

A repaired hypothesis must remain provisional until independently checked.
Repair must not silently overwrite the evidence that invalidated its
predecessor.

## Artifacts

Scripts are under autonomous_lab/. Detailed results remain local under
autonomous_lab/runs/, excluded from Git. The companion JSON preserves
configurations, summaries, and source hashes.
"""

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(text, encoding="utf-8")
summary_path.write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"Cree : {destination}")
print(f"Cree : {summary_path}")
