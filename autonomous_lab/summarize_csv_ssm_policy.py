import hashlib
import json
from pathlib import Path

document = Path("docs/csv-ssm-test-selection.md")
summary_path = Path("docs/csv-ssm-test-selection.json")
if document.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

sources = {
    "training_catalog_audit": ("csv_policy_training_audit_01", "standard"),
    "teacher_initial": ("csv_symbolic_teacher_01", "standard"),
    "dataset_initial": ("csv_policy_dataset_01", "flat"),
    "lookup_initial": ("csv_lookup_baseline_01", "standard"),
    "student_initial": ("csv_ssm_policy_01", "training"),
    "rollout_initial": ("csv_policy_rollout_01", "standard"),
    "teacher_branch_control": ("csv_teacher_branch_01", "standard"),
    "teacher_branched": ("csv_branched_teacher_01", "standard"),
    "dataset_branched": ("csv_policy_dataset_02", "flat"),
    "student_branched": ("csv_ssm_policy_02", "training"),
    "rollout_branched": ("csv_policy_rollout_02", "standard"),
}
summary = {}
documents = {}

for label, (folder, kind) in sources.items():
    path = Path("autonomous_lab/runs") / folder / "results.json"
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    documents[label] = data
    entry = {"source_sha256": hashlib.sha256(raw).hexdigest()}
    if kind == "flat":
        entry["summary"] = data
    else:
        for key in ("config", "summary", "summaries"):
            if key in data:
                entry[key] = data[key]
        if kind == "training":
            entry["results"] = data["results"]
    summary[label] = entry

rollout = documents["rollout_branched"]["results"]
table = []
for policy in ("lookup", "symbolic", "ssm_97001", "ssm_97002", "ssm_97003"):
    rows = [
        row for row in rollout
        if row["policy"] == policy and row["group"] != "known"
    ]
    table.append(
        f"| {policy} | "
        f"{sum(row['detected_unknown'] for row in rows)}/{len(rows)} | "
        f"{sum(row['calls'] for row in rows)/len(rows):.3f} | "
        f"{sum(row['rollout_microseconds'] for row in rows)/len(rows):.1f} |"
    )

text = """# SSM selection of CSV tests

## Scope

A supervised SSM selects predefined black-box tests.
An independent symbolic verifier eliminates incompatible CSV configurations.

The model receives test history features, the known-configuration
candidate mask, and remaining budgets. It does not interpret raw CSV
outputs directly or invent new probes.

Budgets: eight calls, 24 cells, and 512 UTF-8 bytes of cell text.

## Training and evaluation behaviors

Training uses individual programmed transformations.
Evaluation uses four compositions:
- NFC normalization, then right-whitespace stripping.
- Decimal-comma conversion, then row sorting.
- Leading-zero normalization, then empty-column removal.
- Case folding, then row deduplication.

All 48 base CSV configurations are evaluated for each composition.
These are compositions of familiar operations, not new primitives.

## Symbolic reference

The training simulator defines 624 environments but only 600 distinct
response patterns on the catalog. Equivalent aliases are grouped.

The greedy teacher detects 504/552 detectable classes, with no false
rejection of the 48 observable known classes. It misses row deduplication
under its budget despite feasible revealing tests.

The teacher is therefore a reference, not a globally optimal policy.

## Initial imitation and interaction failure

Initial trajectories contain 257 distinct states.
Every validation decision is in a state already present in training.

Both lookup and all three SSMs achieve 462/462 teacher-action matches.

In interaction:
- Initial lookup and teacher detect 192/192 composed environments.
- SSMs detect 184/192, 177/192, and 180/192.

The first divergence in every missed SSM episode selects an action tied
under the teacher score. Training targets permit both actions, but
recorded trajectories cover only one continuation.

Forcing the teacher to take the alternate action and recompute subsequent
decisions still detects 192/192 cases. This supports the diagnosis that
authorized branches were insufficiently covered.

## Branched trajectories

Four randomized tie-breaking trajectories per observable class produce:
- 2,400 trajectories and 1,044 distinct states.
- 1,984 training trajectories and 416 validation trajectories.
- 960 distinct training states.
- 1,730 familiar-state and 118 unseen-state validation decisions.

All three SSMs achieve:
- 8,136/8,136 training matches.
- 1,730/1,730 familiar-state validation matches.
- 118/118 unseen-state validation matches.
- No infeasible predictions.

Training remains supervised imitation. The 118 decisions are related
trajectory states, not 118 independent new behaviors.

## Interaction regression after branched training

| Policy | Composed environments detected | Mean calls | Mean rollout microseconds |
|---|---:|---:|---:|
""" + "\n".join(table) + """

All three SSMs, the new lookup, and the teacher correctly recognize all
48 known configurations without false rejection.

SSMs and teacher consume 8.125 cells per composed episode on average.
The new lookup consumes approximately 8.089.

The new lookup encounters states absent from its table and uses its simple
fallback. Its regression does not invalidate the previous table's success:
the tables memorize different state sets and can choose different tied
actions.

## Timing interpretation

SSMs outperform this uncached teacher implementation in measured rollout
time, but remain about 16 times slower than lookup.

Timing includes policy selection and known-family verification, but excludes
loading, simulator precomputation, and final scoring.

No advantage over an optimized symbolic policy is established.

## Evaluation status

The compositions were initially reserved, then consulted and used to
diagnose failures. The branched-training comparison is therefore a
regression experiment, not independent final validation.

The correction succeeds on these cases without proving general autonomous
learning or reproduction of the unknown tool behavior.

## Decision

Preserve the three trained models and all results.
Do not select only the best seed or retune on these compositions.

Freeze this milestone before an independent evaluation.
The demonstrated capability is pretrained, frozen-weight selection of
useful tests within a programmed catalog and verifier.

## Artifacts

Scripts and small protocol artifacts are under autonomous_lab/.
Detailed runs, trajectories, generated datasets, and model weights remain
local under autonomous_lab/runs/, excluded from Git.

The companion JSON preserves configurations, summaries, per-seed imitation
scores, and hashes of detailed source files.
"""

document.parent.mkdir(parents=True, exist_ok=True)
document.write_text(text.rstrip() + "\n", encoding="utf-8")
summary_path.write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"Cree : {document}")
print(f"Cree : {summary_path}")
