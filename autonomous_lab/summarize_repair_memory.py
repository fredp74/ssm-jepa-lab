import hashlib
import json
from pathlib import Path

source = Path("autonomous_lab/runs/repair_memory_control_01/results.json")
document = Path("docs/local-hypothesis-repair.md")
summary_path = Path("docs/repair-memory-control.json")
marker = "## Control: repair versus observation memory"

text = document.read_text(encoding="utf-8-sig")
if marker in text or summary_path.exists():
    raise SystemExit("Controle deja documente : aucun remplacement.")

raw = source.read_bytes()
data = json.loads(raw.decode("utf-8-sig"))
scores = data["summaries"]["near_miss / repair_provisional"]

cases = scores["cases"]
memory_exact = scores["memory_exact_functions"]
repair_exact = scores["repair_exact_functions"]
gain = scores["repair_correct"] - scores["memory_correct"]

addition = f"""

{marker}

A retrospective matched control compares accepted repairs with the original
rule overridden by every observed input-output pair. Both methods are
evaluated on exactly the same unqueried inputs.

For the {cases} accepted near-miss repairs:

| Method | Exact functions | Correct unqueried outputs |
|---|---:|---:|
| Original rule + observation memory | {memory_exact}/{cases} | {scores['memory_correct']}/{scores['unqueried_inputs']} |
| Repaired rule | {repair_exact}/{cases} | {scores['repair_correct']}/{scores['repair_evaluated_inputs']} |

Observation memory alone fully corrects {memory_exact} functions.
Rule search adds exact recovery for {repair_exact-memory_exact} functions
and corrects {gain} additional unqueried outputs.

The accepted random outside-family repair remains at 6/15, identical
to observation memory.

This narrows the interpretation of the earlier result: not all successful
repairs require generalization beyond stored observations.

Limitations:
- Retrospective analysis of the same cases.
- Accepted repairs are selected outcomes, not all attempted repairs.
- No additional environment queries.
- The repair family was designed for single-output exceptions.
- Exactness is determined by the evaluator, not certified by the agent.

The companion summary is repair-memory-control.json.
"""

document.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
summary_path.write_text(
    json.dumps({
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "config": data["config"],
        "summaries": data["summaries"],
    }, indent=2),
    encoding="utf-8",
)
print(f"Mis a jour : {document}")
print(f"Cree : {summary_path}")
