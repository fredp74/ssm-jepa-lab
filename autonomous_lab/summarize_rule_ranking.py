import hashlib
import json
import statistics
from pathlib import Path

document = Path("docs/ssm-rule-ranking.md")
summary_path = Path("docs/ssm-rule-ranking.json")
if document.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

paths = {
    "ranking": Path("autonomous_lab/runs/rule_ranking_01/results.json"),
    "timing": Path("autonomous_lab/runs/rule_ranking_timing_01/results.json"),
}
data = {}
summary = {}
for name, path in paths.items():
    raw = path.read_bytes()
    data[name] = json.loads(raw.decode("utf-8-sig"))
    summary[name] = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "config": data[name]["config"],
    }

summary["ranking"]["results"] = [
    {"seed": row["seed"], "summaries": row["summaries"]}
    for row in data["ranking"]["results"]
]
summary["timing"]["results"] = data["timing"]["results"]

ranking_table = []
for count in ("4", "8", "12"):
    rows = [
        result["summaries"][count]
        for result in data["ranking"]["results"]
    ]
    fixed = statistics.mean(
        row["fixed"]["mean_candidates_checked"] for row in rows
    )
    guided = statistics.mean(
        row["ssm"]["mean_candidates_checked"] for row in rows
    )
    accuracy = {}
    for method in ("fixed", "ssm"):
        correct = sum(row[method]["correct_unqueried"] for row in rows)
        total = sum(row[method]["total_unqueried"] for row in rows)
        accuracy[method] = 100*correct/total
    ranking_table.append(
        f"| {count} | {fixed:.2f} | {guided:.2f} | "
        f"{accuracy['fixed']:.2f}% | {accuracy['ssm']:.2f}% |"
    )

times = {}
for method in (
    "symbolic_python",
    "symbolic_vectorized",
    "ssm_ranking_and_verification",
):
    times[method] = statistics.mean(
        row["median_microseconds"]
        for row in data["timing"]["results"]
        if row["method"] == method
    )
ratio = times["ssm_ranking_and_verification"]/times["symbolic_python"]

text = f"""# SSM-guided rule ranking: search reduction versus runtime

## Question

Can a trained SSM accelerate symbolic hypothesis search by ranking
candidates before an independent consistency verifier examines them?

## Protocol

The programmed family contains 137 Boolean functions on five input bits.
All rules appear in training. Observation contexts are split into
120 training contexts and 30 validation contexts.

Training targets distribute probability uniformly across every rule
compatible with the observations. The SSM learns from symbolic supervision;
it does not invent the hypothesis family.

Three models use seeds 91011, 91012, and 91013.

At evaluation, both methods stop at the first candidate consistent with
every observation. Compatibility does not establish the true rule.

## Ranking results

Means across three seeds:

| Observations | Fixed candidates checked | SSM candidates checked | Fixed unknown-output accuracy | SSM unknown-output accuracy |
|---|---:|---:|---:|---:|
{chr(10).join(ranking_table)}

The SSM reduces candidate checks consistently. It does not improve
prediction accuracy; with four observations it performs worse.

## CPU timing

The benchmark uses 300 validation episodes, equally divided among
4, 8, and 12 observations. It runs one episode at a time, with one CPU
thread, warmup, and five timing repetitions.

The following values average the per-model median latencies:

| Method | Microseconds per episode |
|---|---:|
| Symbolic Python search | {times['symbolic_python']:.2f} |
| Symbolic vectorized verification | {times['symbolic_vectorized']:.2f} |
| SSM inference + sorting + verification | {times['ssm_ranking_and_verification']:.2f} |

The guided method is approximately {ratio:.1f} times slower than Python
search in this benchmark.

Neural timing includes inference, sorting, and candidate verification.
All timings exclude file loading, model loading, and input preparation.

## Interpretation

Candidate-check reduction is not runtime acceleration. On this small
family, inference overhead outweighs the saved verification work.

Vectorization also costs more than the short Python search here.
This does not establish the same ranking on larger candidate spaces,
other hardware, or expensive verification procedures.

## Limitations

- Programmed family and synthetic training labels.
- Familiar rules, with held-out observation contexts.
- Three training seeds and one machine.
- Shared evaluation cases, not independent datasets per seed.
- Batch-one timing only.
- Different compatible rules may be selected by different methods.
- Fixed candidate order is one baseline, not every possible symbolic method.
- Training cost is additional and excluded from inference timings.

## Decision

Keep symbolic Python search as the practical reference for this task.
Do not claim a neural speedup or a general benefit of SSMs.

A future scaling experiment should specify its search size and verifier
cost before evaluation, rather than expand the problem solely to obtain
a favorable neural result.

## Artifacts

Scripts are under autonomous_lab/. Detailed ranking and timing results
remain local under autonomous_lab/runs/, excluded from Git.

The companion JSON preserves configurations, per-seed summaries, timing
measurements, and source-file hashes.
"""

document.parent.mkdir(parents=True, exist_ok=True)
document.write_text(text.rstrip() + "\n", encoding="utf-8")
summary_path.write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"Cree : {document}")
print(f"Cree : {summary_path}")
