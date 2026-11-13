import hashlib
import json
from pathlib import Path

result_path = Path("autonomous_lab/runs/csv_prospective_01/results.json")
evaluator_path = Path("autonomous_lab/evaluate_csv_prospective.py")
document = Path("docs/csv-prospective-evaluation.md")
summary_path = Path("docs/csv-prospective-evaluation.json")

if document.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

raw = result_path.read_bytes()
data = json.loads(raw.decode("utf-8-sig"))

environments = {
    (row["group"], row["configuration"])
    for row in data["results"] if row["group"] != "known"
}
actual_count = len(environments)
assert actual_count == 576
assert len(data["config"]["reserved_compositions"]) == 12

# Corriger le code, sans modifier les resultats bruts.
source = evaluator_path.read_text(encoding="utf-8-sig")
old = '"reserved_composition_environments": 192,'
new = (
    '"reserved_composition_environments": '
    'len(protocol["reserved_behavior_compositions"])*len(configs),'
)
if source.count(old) == 1:
    corrected_source = source.replace(old, new, 1)
elif source.count(new) == 1:
    corrected_source = source
else:
    raise SystemExit("Compteur de l'evaluateur inattendu : aucune modification.")
compile(corrected_source, str(evaluator_path), "exec")

summaries = []
table = []
for policy in data["config"]["policies"]:
    rows = [
        row for row in data["results"]
        if row["policy"] == policy and row["group"] != "known"
    ]
    known = [
        row for row in data["results"]
        if row["policy"] == policy and row["group"] == "known"
    ]
    summary = {
        "policy": policy,
        "composed_episodes": len(rows),
        "detected": sum(row["detected_unknown"] for row in rows),
        "incorrect_survivors": sum(
            not row["detected_unknown"] and row["exact"] < row["cases"]
            for row in rows
        ),
        "known_control_episodes": len(known),
        "known_false_rejections": sum(row["detected_unknown"] for row in known),
        "known_fully_correct": sum(row["exact"] == row["cases"] for row in known),
        "mean_calls": sum(row["calls"] for row in rows)/len(rows),
        "mean_cells": sum(row["cells"] for row in rows)/len(rows),
        "mean_rollout_microseconds": sum(
            row["rollout_microseconds"] for row in rows
        )/len(rows),
        "unfamiliar_state_decisions": sum(
            row["unfamiliar_state_decisions"] for row in rows
        ),
    }
    summaries.append(summary)
    table.append(
        f"| {policy} | {summary['detected']}/{len(rows)} | "
        f"{100*summary['detected']/len(rows):.2f}% | "
        f"{summary['mean_calls']:.3f} | "
        f"{summary['mean_rollout_microseconds']:.1f} |"
    )

erratum = {
    "field": "config.reserved_composition_environments",
    "recorded_value": data["config"]["reserved_composition_environments"],
    "correct_value": actual_count,
    "raw_result_modified": False,
    "effect_on_scores": "none",
    "future_evaluator_counter_corrected": True,
}

text = """# Prospective CSV composition evaluation

## Protocol

The protocol was frozen before execution using seed 20261113.
Model, lookup, teacher, and catalog fingerprints were checked.

The test contains:
- 12 ordered compositions of three familiar operations.
- All 48 base configurations per composition: 576 environments.
- 48 known-configuration controls.
- 100 newly generated evaluation tables.

Budgets remain eight calls, 24 cells, and 512 UTF-8 bytes.
All three prespecified SSMs are evaluated without parameter updates.

This is a prospective test within the same synthetic domain, not an
independent blind evaluation or discovery of new transformation primitives.

## Results

| Policy | Detected / episodes | Detection rate | Mean calls | Mean rollout microseconds |
|---|---:|---:|---:|---:|
""" + "\n".join(table) + """

The lookup, teacher, and each SSM detect all 576 composed environments.
They correctly recognize all 48 known configurations without false rejection.

For random selection, five paths are evaluated per environment.
Its 2,880 composed episodes are not 2,880 independent environments.

## What transfers

The composed behaviors are new, but lookup, teacher, and SSMs encounter
no decision states absent from the branched training-state table.

The result demonstrates reuse of a pretrained testing policy on these
new compositions. It does not demonstrate generalization to new decision
states, interpretation of arbitrary CSV semantics, or reproduction of
unknown tool behavior.

The evaluation tables provide witnesses distinguishing every tested
composition from the known family. Matching on a finite table set would
not by itself establish global behavioral equivalence.

## Practical comparison

Lookup remains the efficient reference: approximately 49 microseconds per
composed episode, versus 660-779 microseconds for the SSMs.

Teacher timing reflects an uncached implementation.
Timings include selection and known-family verification, but exclude
loading, simulator precomputation, and final scoring.
They are indicative measurements from this run, not a universal benchmark.

No SSM advantage over lookup is demonstrated.

## Metadata erratum

The raw result retains an inherited value of 192 in
config.reserved_composition_environments.

The actual data contains 12 x 48 = 576 distinct composed environments.
This accounting error does not change scores or evaluated cases.

The raw file is preserved. The companion summary records the correction
and its source hash. The evaluator code now computes the count dynamically.

## Limitations and decision

- Familiar programmed operations and catalog.
- Handcrafted simulator and symbolic observation features.
- Related configurations within each composition.
- All three seeds share evaluation environments.
- No new decision states for the successful adaptive policies.
- Detection is distinct from learning to reproduce the tool.

Freeze this result without retuning on it. The compositions are now
consulted and must be treated as regression cases in future work.

## Artifacts

The frozen protocol is under autonomous_lab/data/csv_prospective_01/.
Detailed local results remain under autonomous_lab/runs/csv_prospective_01/,
excluded from Git.
"""

document.parent.mkdir(parents=True, exist_ok=True)
evaluator_path.write_text(corrected_source, encoding="utf-8")
document.write_text(text.rstrip() + "\n", encoding="utf-8")
summary_path.write_text(
    json.dumps({
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "recorded_config": data["config"],
        "metadata_erratum": erratum,
        "summaries": summaries,
    }, indent=2),
    encoding="utf-8",
)
print(f"Cree : {document}")
print(f"Cree : {summary_path}")
print("Resultat brut conserve ; compteur futur corrige.")
