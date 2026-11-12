import hashlib
import json
from pathlib import Path

source = Path("autonomous_lab/runs/csv_common_budget_01/results.json")
document = Path("docs/csv-common-budget.md")
summary_path = Path("docs/csv-common-budget.json")
if document.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

raw = source.read_bytes()
data = json.loads(raw.decode("utf-8-sig"))

table = []
for row in data["summaries"]:
    if row["group"] == "additional":
        table.append(
            f"| {row['plan']} | {row['detected_unknown']}/"
            f"{row['environments']} | {row['incorrect_survivors']} | "
            f"{row['mean_calls']:.3f} | {row['mean_cells']:.1f} |"
        )

text = """# CSV behavior detection under a common budget

## Question

How do cell contrasts, structural relations, and a combined plan compare
under the same maximum interaction and payload budgets?

The plans were frozen before this evaluation. No neural learning or
SSM/JEPA mechanism is involved.

## Protocol

All plans allow at most:
- Eight calls.
- 24 cells.
- 512 UTF-8 bytes of cell text.

Transport encoding is excluded from the byte budget.
Execution stops when a contradiction is detected.

The structural and combined plans contain seven calls; the cell-contrast
plan contains eight. Equal upper budgets do not mean identical payloads.

Evaluation covers:
- All 48 known CSV configurations.
- Eight additional handcrafted behaviors on four base configurations.
- 100 newly generated evaluation tables.

This is not an independent blind evaluation.

## Results

All three plans recognize the 48 known configurations correctly,
without false rejection.

Results on the 32 additional environments:

| Plan | Detected | Incorrect survivors | Mean calls | Mean cells |
|---|---:|---:|---:|---:|
""" + "\n".join(table) + """

Structural testing detects duplicate-row removal through the duplication
relation. Neither the contrast nor combined plan includes that relation.

The combined plan trades some structural checks for cell contrasts.
It therefore does not dominate the specialized plans under this budget.

## Shared blind spots

All plans miss:
- Removal of entirely empty columns.
- Escaping spreadsheet-formula-like values.

These behaviors remain incorrect surviving hypotheses, not successful
configuration discoveries.

## Interpretation

Coverage depends on which properties the budget permits testing.
Combining test categories does not guarantee broader effective coverage
when individual checks must be removed.

The structural plan performs best on this constructed behavior set.
This does not establish universal superiority.

## Limitations

- Handwritten plans, behaviors, and relations.
- Eight semantic behaviors, not 32 independent behavior types.
- Different actual byte volumes and call counts.
- Early stopping changes consumed payload across environments.
- No estimate of real-world unknown-behavior frequencies.
- Detection leads to rejection or abstention, not reproduction of the tool.

## Decision

Freeze and preserve these results. Do not repair the two blind spots
and present the revised score as independent validation.

A subsequent experiment may test choosing checks from observations and
remaining budget. That would be an adaptive testing policy, not necessarily
a learned policy.

## Artifacts

The plans are stored in
autonomous_lab/data/csv_common_budget_01/plans.json.

The evaluator is autonomous_lab/evaluate_csv_common_budget.py.
Detailed local results remain under autonomous_lab/runs/csv_common_budget_01/,
excluded from Git.

The companion JSON preserves the configuration, summaries, and source hash.
"""

document.parent.mkdir(parents=True, exist_ok=True)
document.write_text(text.rstrip() + "\n", encoding="utf-8")
summary_path.write_text(
    json.dumps({
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "config": data["config"],
        "summaries": data["summaries"],
    }, indent=2),
    encoding="utf-8",
)
print(f"Cree : {document}")
print(f"Cree : {summary_path}")
