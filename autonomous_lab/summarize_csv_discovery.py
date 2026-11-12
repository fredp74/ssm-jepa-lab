import hashlib
import json
from pathlib import Path

document = Path("docs/csv-tool-discovery.md")
summary_path = Path("docs/csv-tool-discovery.json")
if document.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

runs = {
    "environment": "csv_environment_01",
    "discovery": "csv_discovery_01",
    "unknown_behavior": "csv_unknown_behavior_01",
    "diversity": "csv_diversity_01",
    "contrasts": "csv_contrasts_01",
}
summary = {}

for label, folder in runs.items():
    path = Path("autonomous_lab/runs") / folder / "results.json"
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    entry = {"source_sha256": hashlib.sha256(raw).hexdigest()}

    if label == "environment":
        entry["summary"] = data
    else:
        entry["config"] = data["config"]
        if "summaries" in data:
            entry["summaries"] = data["summaries"]
        else:
            entry["summaries"] = []
            for policy in ("fixed", "active"):
                rows = [
                    row for row in data["results"]
                    if row["policy"] == policy
                ]
                entry["summaries"].append({
                    "policy": policy,
                    "environments": len(rows),
                    "calls": sum(row["calls"] for row in rows),
                    "exact": sum(row["exact"] for row in rows),
                    "answered": sum(row["answered"] for row in rows),
                    "identified": sum(
                        row["remaining_candidates"] == 1 for row in rows
                    ),
                })
    summary[label] = entry

text = """# Discovering an unfamiliar CSV configuration

## Scope

A symbolic agent identifies a black-box CSV configuration by querying it.
The family of possible behaviors and its simulator are programmed.

No neural training, SSM, or JEPA mechanism is used.
The tool interface hides configuration by convention, not security isolation.

## Environment

The initial family contains 48 configurations varying delimiter,
quoting, quote escaping, line endings, and whitespace trimming.

Environment checks passed:
- 4,800 serialization/readback checks.
- Query-budget enforcement.
- Protection of logged inputs against subsequent mutation.

The test score requires exact serialized output, not merely equivalent CSV.

## Configuration discovery and persistence

Across all 48 known configurations:
- Fixed probe order: four calls per configuration.
- Active selection: one call per configuration.
- Both: 4,800/4,800 reserved predictions correct after memory reload.
- No extra tool calls during prediction.

The active policy always selects composite probe 7, which distinguishes
all configurations. A fixed policy using that probe first would match it.

This is configuration identification in a known family, not complex
autonomous learning.

## Unknown behavior

An initial test used 24 outside-family environments:
six handcrafted behaviors on four base configurations.

The probes detected 20/24. They missed leading-zero normalization in all
four cases, producing only 80/100 correct reserved outputs per case.

Trying all eight probes did not fix this blind spot.

## Randomized category diversity

Three frozen category-based probe banks were evaluated on eight additional
handcrafted behaviors, each applied to four base configurations.

All banks retained correct behavior on the 48 known configurations.

| Bank | Additional outside-family cases detected |
|---|---:|
| Fixed | 12/32 |
| Diverse 111001 | 28/32 |
| Diverse 111002 | 20/32 |
| Diverse 111003 | 24/32 |

Only one diversified bank detected the known leading-zero regression.
Category labels alone did not guarantee a revealing example was sampled.

## Deterministic contrasts: regression result

Contrasts were constructed after inspecting the observed failures.

| Measurement | Fixed bank | Contrast bank |
|---|---:|---:|
| Known configurations correct | 48/48 | 48/48 |
| Leading-zero regression detected | 0/4 | 4/4 |
| Additional outside-family cases detected | 12/32 | 32/32 |
| Incorrect surviving hypotheses across outside groups | 24 | 0 |

Both banks have a maximum of eight calls and 24 cells.
Input text volume is 205 UTF-8 bytes for fixed probes and 166 for contrasts.

On the 32 additional outside-family cases, detection uses 5.875 calls
on average for the fixed bank and 4.5 for contrasts. Calls stop on
contradiction, so actual consumed input volume varies by environment.

## Interpretation

Designed contrasts cover the diagnosed failure modes.
This is a regression success, not independent evidence of general
unknown-behavior detection.

Detection eliminates incompatible hypotheses and leads to abstention.
It does not reproduce the newly encountered behavior.

## Limitations

- Handwritten family, probes, transformations, and evaluator.
- Small synthetic tables and handcrafted outside-family behaviors.
- Repeated use of diagnostic behaviors and evaluation cases.
- Four base configurations per outside behavior are related cases,
  not four independent semantic behaviors.
- Equal cell budgets do not imply equal byte volumes.
- Successful probes do not certify behavior on arbitrary future inputs.

## Decision

Freeze the deterministic contrast bank.
Do not modify it in response to the next reserved evaluation.

Future work should evaluate new behaviors separately from known
regressions, and report detection, abstention, reproduction accuracy,
and interaction cost as distinct measurements.

## Artifacts

Scripts are under autonomous_lab/. Small frozen probe banks are under
autonomous_lab/data/. Detailed results and reserved cases remain local
under autonomous_lab/runs/, excluded from Git.

The companion JSON retains configurations, summaries, and source hashes.
"""

document.parent.mkdir(parents=True, exist_ok=True)
document.write_text(text.rstrip() + "\n", encoding="utf-8")
summary_path.write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"Cree : {document}")
print(f"Cree : {summary_path}")
