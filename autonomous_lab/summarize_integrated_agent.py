import hashlib
import json
from pathlib import Path

document = Path("docs/integrated-agent-and-freshness.md")
summary_path = Path("docs/integrated-agent-and-freshness.json")
if document.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

runs = {
    "integrated_initial": "integrated_agent_01",
    "integrated_corrected": "integrated_agent_02",
    "audit_coverage": "audit_coverage_01",
    "change_recovery": "change_recovery_01",
    "adaptive_recovery": "adaptive_recovery_01",
    "history_reliability": "history_reliability_01",
    "reliability_gate": "reliability_gated_recovery_01",
}
summary = {}
documents = {}

for label, folder in runs.items():
    path = Path("autonomous_lab/runs") / folder / "results.json"
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    documents[label] = data
    entry = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "config": data["config"],
    }
    for key in ("summary", "summaries", "coverage_check", "bounds"):
        if key in data:
            entry[key] = data[key]
    summary[label] = entry

gate_table = []
for threshold in (0.1, 0.2, 0.4, 0.6):
    rows = [
        row for row in documents["reliability_gate"]["summaries"]
        if row["threshold"] == threshold
    ]
    answered = sum(row["answered"] for row in rows)
    total = sum(row["evaluated_inputs"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    accuracy = f"{100*correct/answered:.2f}%" if answered else "N/A"
    gate_table.append(
        f"| {100*threshold:.0f}% | {100*answered/total:.2f}% | {accuracy} |"
    )

text = """# Integrated agent, change detection, and memory freshness

## Scope

A symbolic prototype integrates persistent observations, provisional
hypotheses, contradiction detection, grammar expansion, and epoch archives.

No neural model, SSM, or JEPA objective is used in these experiments.
The hypothesis and repair families are programmed.

## Integration demonstration

The initial demonstration ran 96 interactions with an unannounced
environment inversion after interaction 48.

- Six save/reload checks preserved state.
- The change was detected at interaction 49.
- The previous epoch was archived.
- Final predictions were correct on all 32 inputs.

All inputs had been observed by the end. Final accuracy therefore
establishes recall, not held-out generalization.

## Audit correction

The initial audit unnecessarily repeated inputs.
A persistent round-robin cursor corrected this implementation defect.

Testing every single-change position from the same fully observed state:

| Audit | Mean detection delay | Maximum delay | Detected within 32 interactions |
|---|---:|---:|---:|
| Original | 63 | 125 | 8/32 |
| Corrected | 16.5 | 32 | 32/32 |

This is a software correction, not learned behavior. The bound applies
to the fully observed, deterministic setting tested.

## Recovery after change

Two strategies received identical post-detection observations:
reset the model or retain the old table with observed entries corrected.

Historical values help strongly after small changes but fail after a
global inversion. Resetting can recover from inversion, yet many other
modified functions fall outside its repair grammar and cause abstention.

A handwritten adaptive policy switched based on the mismatch fraction.
It handled the extremes but sometimes degraded performance on intermediate
changes. Its threshold was not adopted as a general solution.

## Reliability-aware abstention

A subsequent control separated:
- reliability of historical outputs;
- reliability of a frozen replacement hypothesis.

After detection, eight observations fit the replacement and eight
independent holdout observations check it. History uses all 16 checks.
Both methods are evaluated on the same 15 remaining inputs.

Hypergeometric upper bounds use alpha 0.025 for each proposal.
By a union bound, both bounds cover simultaneously with at least 95%
confidence for a fixed case and fixed budget, under the sampling assumptions.

This does not guarantee conditional accuracy among accepted predictions,
nor validity under repeated optional stopping.

| Allowed upper error fraction | Coverage | Accuracy among answered inputs |
|---|---:|---:|
""" + "\n".join(gate_table) + """

At tolerance 40%, only history was selected.
At tolerance 60%, replacement was selected in three cases.

The tolerance values were prespecified for a trade-off study.
No threshold is selected as a deployment setting from these results.

## Interpretation

A system can preserve evidence and explicitly abstain when checks do not
justify a response. Recognizing stale history does not establish that
a replacement model is reliable.

Strict tolerances can require more evidence than the available budget.
Abstention does not imply that every proposed answer would be wrong.

## Limitations

- Small deterministic domain, without noise.
- Handwritten policies and programmed hypothesis families.
- One inversion scenario; sampled intermediate changes.
- Several experiments reuse the same initial state and related scenarios.
- Some recovery outcomes are constrained by representation mismatch.
- Aggregate coverage and accuracy reflect the constructed scenario mixture.
- The agent has not demonstrated general autonomous SSM/JEPA learning.

## Decision

Keep the corrected audit cursor.
Keep historical observations with explicit epoch provenance.
Do not treat historical values or compatible hypotheses as certified truth.
Do not adopt the adaptive threshold based on this diagnostic benchmark.

Before further design changes, specify the acceptable error criterion,
interaction budget, and assumptions about environmental change.

## Artifacts

Scripts are under autonomous_lab/. Detailed runs and checkpoints remain
local under autonomous_lab/runs/, excluded from Git.

The companion JSON preserves configurations, summaries, mathematical
coverage checks, and source hashes.
"""

document.parent.mkdir(parents=True, exist_ok=True)
document.write_text(text.rstrip() + "\n", encoding="utf-8")
summary_path.write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"Cree : {document}")
print(f"Cree : {summary_path}")
