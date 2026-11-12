import json
from pathlib import Path

root = Path("autonomous_lab/runs")
destination = Path("docs/exploration-replications.md")
summary_path = Path("docs/exploration-replications.json")

if destination.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

results = []
for number in ("01", "02", "03"):
    path = root / f"exhaustive_ensemble_random_{number}" / "results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    comparisons = {
        row["policy"]: row for row in data["comparisons"]
    }
    results.append({
        "ensemble": number,
        "seeds": data["config"]["ensemble_seeds"],
        "random_exhaustive": data["summary"]["accuracy"],
        "entropy": comparisons["entropy"],
        "disagreement": comparisons["disagreement"],
    })

table = "\n".join(
    f"| {row['ensemble']} "
    f"| {100*row['random_exhaustive']:.2f}% "
    f"| {100*row['entropy']['accuracy']:.2f}% "
    f"| {100*row['disagreement']['accuracy']:.2f}% |"
    for row in results
)

entropy_gain = sum(
    row["entropy"]["difference_percentage_points"] for row in results
) / len(results)
disagreement_gain = sum(
    row["disagreement"]["difference_percentage_points"] for row in results
) / len(results)

text = f"""# Sequential exploration: three-ensemble replication

## Question

Can a frozen neural ensemble choose more useful questions than random
sampling in the three-bit affine-rule environment?

## Protocol

Three independently trained ensembles contain three diagonal SSMs each.
Training includes zero to four observations per episode.

Each evaluation episode allows four distinct questions. Final answers
use the mean probabilities of the same ensemble for every policy.

Policies:
- Random sampling without replacement.
- Maximum predictive entropy of the ensemble mean.
- Ensemble disagreement: entropy of the mean minus mean member entropy.

No model parameters are updated during evaluation.

The random reference exhaustively covers 70 question sets, 24 orders,
and all 16 rules. Guided policies use 24 repetitions per rule, including
random tie-breaking where applicable.

## Results

| Ensemble | Exhaustive random | Entropy | Disagreement |
|---|---:|---:|---:|
{table}

Mean difference from each ensemble's exhaustive random reference:
- Entropy: {entropy_gain:+.2f} percentage points.
- Disagreement: {disagreement_gain:+.2f} percentage points.

Neither guided policy improves on random for every ensemble.

## Interpretation

The first ensemble suggested a benefit, but replication showed substantial
dependence on the trained ensemble. A positive average across three
ensembles does not establish a reliable improvement.

Predictive uncertainty and ensemble disagreement are not guaranteed
measures of information gain about the hidden rule.

## Limitations

- One small, familiar family of 16 rules.
- Only three ensembles.
- Shared evaluation cases; counts are not independent trials.
- Guided-policy repetitions can repeat identical trajectories.
- Random performance is exhaustive; guided-policy tie-breaking is sampled.
- Evaluation includes both training and held-out four-question contexts.
- The benchmark has been consulted repeatedly during development.
- Prior neural training is required; this is not learning from nothing.

## Decision

Close this experiment without selecting a guided policy as a demonstrated
improvement. Preserve the protocol and results before changing the design.

The system supports interaction-driven state adaptation, but has not
demonstrated robust autonomous acquisition of new rule families.

## Artifacts

Scripts are in autonomous_lab/. Detailed local results and checkpoints
are under autonomous_lab/runs/ and remain excluded from Git.
"""

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(text, encoding="utf-8")
summary_path.write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)
print(f"Cree : {destination}")
print(f"Cree : {summary_path}")
