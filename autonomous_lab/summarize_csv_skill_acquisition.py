import hashlib
import json
from pathlib import Path

source = Path("autonomous_lab/runs/csv_skill_acquisition_01/results.json")
document = Path("docs/csv-executable-skill-acquisition.md")
summary_path = Path("docs/csv-executable-skill-acquisition.json")

if document.exists() or summary_path.exists():
    raise SystemExit("Bilan deja present : aucun remplacement.")

raw = source.read_bytes()
data = json.loads(raw.decode("utf-8-sig"))
accepted = [
    row for row in data["results"] if row["status"] == "provisional"
]
matched = {
    "accepted_environments": len(accepted),
    "baseline_exact": sum(row["baseline_exact"] for row in accepted),
    "acquired_exact": sum(row["acquired_exact"] for row in accepted),
    "evaluation_cases": sum(row["evaluation_cases"] for row in accepted),
}

text = """# Executable CSV skill acquisition

## Scope

A symbolic agent identifies and stores an executable program consisting
of a programmed transformation followed by CSV serialization.

The candidate language contains 624 programs and 600 distinct response
classes on the probe catalog. Its operations are supplied in advance.

This is program identification within a declared language, not invention
of new operations. No neural training, SSM, or JEPA objective is involved.

## Protocol

The experiment covers 52 environments: identity and 12 individual
transformations, each under four CSV configurations.

Maximum interaction budget:
- Four discovery calls.
- Four distinct verification calls.
- Eight calls, 24 cells, and 512 UTF-8 bytes overall.

The proposal is frozen before verification. A surviving program is stored
with observations and checks, then reloaded before predicting 100
evaluation tables. Prediction makes no additional environment calls.

A stored program remains provisional.

## Results

| Measurement | Result |
|---|---:|
| Environments with an accepted provisional program | 26/52 |
| Environments left ambiguous | 26/52 |
| Exact predictions after program reload | 2,600/2,600 |
| Known-CSV baseline on the same accepted cases | 1,369/2,600 |
| Mean calls across all environments | 5.19 |

The acquired-program coverage is 50%.
Its 100% measured accuracy is conditional on those covered cases.

The known-CSV baseline answers every case and scores 3,538/5,200 overall.
It receives the same observations and selects the original CSV format
matching the largest number of responses. It is not guaranteed consistent
with every observation.

The conditional and overall scores must not be treated as directly
interchangeable measures of superiority.

## Remaining ambiguity

All four identity environments remain ambiguous, as do all tested NFC,
boolean-normalization, deduplication, and empty-column-removal environments.

The agent requires one remaining catalog-response class before storing
a program. This can be unnecessarily restrictive for individual queries:
different hypotheses can agree on a particular requested output.

However, agreement on a finite catalog does not establish global program
equivalence. Ambiguity must not simply be removed by declaring programs
equivalent or relaxing acceptance after inspecting evaluation answers.

## Interpretation

The integrated path succeeds for the covered environments:
interaction -> program selection -> distinct checks -> persistence ->
execution on new tables.

The repertoire is programmed and the tested tools are representable
within it. This does not demonstrate arbitrary tool learning or robust
out-of-language behavior.

## Limitations

- Handwritten program language and query-selection method.
- Four related configurations per transformation.
- One query-selection protocol and one evaluation-case sample.
- Abstention on half the environments.
- Four successful checks are not a general correctness proof.
- No outside-language tools in this acquisition experiment.
- No independent estimate of real-world performance.

## Decision

Preserve this baseline before modifying acceptance or probe selection.

Next, analyze ambiguous cases to separate observational equivalence from
insufficient exploration. Any new candidate-agreement prediction mechanism
must report coverage and accuracy separately.

## Artifacts

The implementation is autonomous_lab/acquire_csv_skills.py.
The executable representation and persistence helpers are in
autonomous_lab/csv_executable_skills.py.

Programs, observations, and detailed local results are under
autonomous_lab/runs/csv_skill_acquisition_01/, excluded from Git.
The companion JSON preserves configuration, summaries, matched-case
baseline scores, and the source hash.
"""

document.parent.mkdir(parents=True, exist_ok=True)
document.write_text(text.rstrip() + "\n", encoding="utf-8")
summary_path.write_text(
    json.dumps({
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "config": data["config"],
        "summary": data["summary"],
        "matched_accepted_cases": matched,
        "per_environment_results": data["results"],
    }, indent=2),
    encoding="utf-8",
)
print(f"Cree : {document}")
print(f"Cree : {summary_path}")
