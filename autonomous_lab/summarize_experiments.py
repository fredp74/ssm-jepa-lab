import hashlib
import json
from pathlib import Path

sources = {
    "exact_memory_mlp": Path(
        "autonomous_lab/runs/context_holdout_01/results.json"
    ),
    "ssm": Path(
        "autonomous_lab/runs/context_holdout_ssm_01/results.json"
    ),
    "ssm_latent": Path(
        "autonomous_lab/runs/context_holdout_latent_01/results.json"
    ),
}
destination = Path("docs/autonomous-learning-baseline.md")
summary_path = Path("docs/autonomous-learning-summary.json")
if destination.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

documents = {}
summary = {}
for name, path in sources.items():
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    documents[name] = data
    summary[name] = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol": data["protocol"],
        "seeds": [
            {
                "seed": row["seed"],
                "train": row["train"]["summary"],
                "test": row["test"]["summary"],
            }
            for row in data["results"]
        ],
    }

reference = documents["ssm"]["protocol"]
for name, data in documents.items():
    for key in (
        "train_contexts", "test_contexts", "rules", "seeds",
        "steps", "batch_size", "learning_rate",
    ):
        if data["protocol"][key] != reference[key]:
            raise SystemExit(f"Protocol mismatch: {name}, {key}")

table = []
for name, label in (
    ("exact_memory_mlp", "Exact memory + MLP"),
    ("ssm", "Diagonal SSM"),
    ("ssm_latent", "SSM + latent objective"),
):
    rows = summary[name]["seeds"]
    correct = sum(row["test"]["correct_unknown"] for row in rows)
    total = sum(row["test"]["total_unknown"] for row in rows)
    table.append(
        f"| {label} | {correct}/{total} | {100*correct/total:.2f}% |"
    )

text = """# Autonomous learning: baseline experiments

## Scope

The long-term goal is an agent that acquires useful knowledge through
interaction. These experiments test small components of that goal.
They do not constitute a language model or a generally autonomous learner.

Learning without manual labels is distinct from learning without training.
The neural experiments below use explicit optimization during development.
Their parameters remain frozen during evaluation episodes.

## Environment and reference

An environment maps three input bits to one output bit using an affine
binary rule. There are 16 possible rules.

A symbolic agent knows this family in advance. With four informative
questions, it identifies every rule. Across all four-question subsets,
random question selection achieves 90% expected held-out accuracy.

This is a programmed reference, not a learned reasoning procedure.

## Early findings

- Neural models trained from scratch within an episode nearly memorized
  queried answers but did not reliably infer unqueried answers.
- Excluding entire rules from training introduced a strong prior mismatch.
  A symbolic reasoner restricted to the training rules also failed on
  excluded rules or found the observations incompatible.
- Simple recurrent state showed recency effects. Initializing retention
  near 0.95 reduced that imbalance without solving generalization.
- Exact keyed memory guaranteed recall but did not by itself solve inference.

## Context holdout protocol

All 16 rules appear during training. Four-question contexts are split
before training into 50 training contexts and 20 held-out contexts.

Both partitions contain 80% informative and 20% ambiguous contexts.
The symbolic reference is therefore 90%, rather than 100%.

Neural runs use three seeds, 1,500 updates, batch size 32, and Adam
with learning rate 0.001. The loss targets unobserved queries.
No parameter updates occur during evaluation.

## Held-out context results

| System | Correct unknown answers | Accuracy |
|---|---:|---:|
""" + "\n".join(table) + """

The diagonal SSM transfers partially to new observation contexts.
Exact memory + MLP scores slightly higher in this experiment, but this
comparison changes the representation and decoder as well as memory.

The latent-objective variant corrects 92 baseline errors and introduces
90 new errors: a net gain of two predictions across the three seeds.
This does not establish a useful improvement.

## Latent objective

The student receives four observations. An EMA teacher receives all eight
input-output pairs during training. A predictor matches normalized teacher
states, with an additional variance penalty.

Weights: task loss 1.0, latent loss 0.1, variance loss 0.01.
Teacher EMA coefficient: 0.99.

This is a JEPA-inspired experiment, not a reproduction of I-JEPA or V-JEPA.
It adds parameters and compute. Representation-collapse diagnostics were
not recorded in the result files, so non-collapse has not been established.

## Interpretation and limitations

- All neural systems received prior training.
- Held-out contexts use a familiar rule family and a tiny input space.
- Exact recall is programmed in the keyed-memory system.
- Observation recall was not part of the SSM training loss in this protocol.
- Seeds share evaluation cases; pooled counts are not independent trials.
- The context split has been consulted repeatedly and now serves as a
  diagnostic benchmark, not an untouched final test.
- No general advantage of SSMs or JEPA is claimed.

## Next question

Can informative question selection improve a frozen neural agent compared
with random selection?

A symbolic information-gain policy will be an explicitly privileged
reference. It must not be presented as a learned neural exploration policy.

## Artifacts

Scripts are in autonomous_lab/. Local runs contain protocols, detailed
results, memories, and checkpoints. Model files and run directories are
excluded from Git.

The companion JSON summary retains per-seed scores, protocols, and hashes
of the detailed source result files.
"""

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(text, encoding="utf-8")
summary_path.write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"Cree : {destination}")
print(f"Cree : {summary_path}")
