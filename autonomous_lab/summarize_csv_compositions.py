import hashlib
import json
from pathlib import Path


def main():
    source = Path("autonomous_lab/runs/csv_composition_01/results.json")
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    docs = Path("docs")
    md = docs / "csv-composition-acquisition.md"
    js = docs / "csv-composition-acquisition.json"

    if md.exists() or js.exists():
        raise SystemExit("Documentation existante : aucun remplacement.")

    summaries = {row["arm"]: row for row in data["summary"]}
    single = summaries["single"]
    composed = summaries["composed"]

    text = f"""# CSV composition acquisition: development pilot

Expanding the candidate language from at most one operation to at most
two ordered operations increased answer coverage from
{single['coverage']:.1%} to {composed['coverage']:.1%} in this development
pilot. The expanded repertoire used more tool calls.

## Experimental scope

The benchmark contains 24 simulated environments: six compositions
across four CSV configurations.

The elementary operations are implemented by humans. Candidate generation
exhaustively enumerates zero, one or two distinct operations in order,
combined with CSV configurations. The expanded language contains 6,960
candidate programs.

This experiment measures selection and retention of composed programs
within that language. It does not demonstrate invention of new primitive
operations, open-ended program synthesis or autonomous language learning.

Some compositions were explored in earlier experiments. This is a
development benchmark, not an evaluation on demonstrably unseen
behavior families.

## Procedure

Both arms start without observations of the current environment.

Discovery selects catalogue probes to reduce candidate uncertainty.
The total budget is eight calls, 24 cells and 512 input-text UTF-8 bytes
per environment. Discovery reserves resources for two verification calls.

Candidate programs are frozen before verification. A provisional status
requires unanimous predictions matching both verification responses.
Disagreement between candidates causes abstention; a response outside
their predictions revokes the proposal.

During evaluation, only provisional memories may answer. Every retained
program must agree on the answer to the current input.

A set of 200 evaluation tables is shared across the 24 environments.
Evaluation feedback is not used to select tests or candidate programs.

## Results

| Metric | At most one operation | At most two operations |
|---|---:|---:|
| Environments | {single['environments']} | {composed['environments']} |
| Provisional environments | {single['statuses'].get('provisional', 0)} | {composed['statuses'].get('provisional', 0)} |
| Answers provided | {single['answered']:,} | {composed['answered']:,} |
| Correct answers | {single['correct']:,} | {composed['correct']:,} |
| Incorrect answers | {single['errors']} | {composed['errors']} |
| Evaluation questions | {single['evaluation_cases']:,} | {composed['evaluation_cases']:,} |
| Answer coverage | {single['coverage']:.1%} | {composed['coverage']:.1%} |
| Tool calls | {single['total_calls']} | {composed['total_calls']} |
| Cells | {single['total_cells']} | {composed['total_cells']} |
| Input-text UTF-8 bytes | {single['total_text_bytes']:,} | {composed['total_text_bytes']:,} |

The composed arm answered {composed['answered']:,} questions with zero
observed errors. It remained verification-ambiguous in 16 environments.

Its accepted environments involved leading-zero normalization followed
by row sorting, and decimal-comma conversion followed by row sorting.

The single-operation arm passed verification in four environments but
made {single['errors']} evaluation errors. Passing two checks is therefore
not a general correctness guarantee.

The arms share resource ceilings, but actual consumption differs.
These results do not establish better performance at equal actual cost.

## Computation

Shared catalogue precomputation took
{data['shared_catalogue_precomputation_seconds']:.3f} seconds on the
reported machine.

Search and verification took
{single['search_and_verification_seconds']:.3f} seconds in the
single-operation arm and
{composed['search_and_verification_seconds']:.3f} seconds in the composed
arm.

These are descriptive measurements from one run, not replicated
performance benchmarks.

## Persistence limitation

Evaluation predictions use programs loaded from saved memory files.
However, the reported memory_roundtrip_passed field compares two reads
of the same file. It does not compare predictions before saving and
after reloading.

That field must not be interpreted as a valid prediction-persistence
test. A corrected experiment should compare both the in-memory payload
and predictions before and after serialization. Raw results are preserved.

## Interpretation

The expanded repertoire supports some behaviors that the initial
repertoire cannot represent adequately. Acquisition remains partial:
16 of 24 composed-arm environments are not accepted.

Zero observed errors in the composed arm must be interpreted within the
simulated candidate family, which contains the true programs. It does
not establish reliability for arbitrary external tools.

The 4,800 questions reuse 200 tables across related environments and are
not 4,800 independent replications.

There is no neural training in this experiment and no demonstrated
SSM/JEPA advantage.

## Reproduction

From the repository root:

1. python autonomous_lab/prepare_csv_composition.py
2. python autonomous_lab/acquire_csv_compositions.py
3. python autonomous_lab/summarize_csv_compositions.py

The experiment scripts refuse to overwrite existing output directories.
Raw results remain local under autonomous_lab/runs/csv_composition_01.
The companion JSON records the source hash, protocol and summary.
"""

    report = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "protocol": data["protocol"],
        "summary": data["summary"],
        "shared_catalogue_precomputation_seconds":
            data["shared_catalogue_precomputation_seconds"],
        "persistence_check_limitation": (
            "memory_roundtrip_passed compares two file reads; "
            "prediction equivalence before and after saving was not tested."
        ),
        "conclusion": (
            "The composed repertoire increased coverage with more calls. "
            "Acquisition remained partial; equal-cost superiority and "
            "open-ended learning were not established."
        ),
    }

    docs.mkdir(exist_ok=True)
    md.write_text(text.rstrip() + "\n", encoding="utf-8")
    js.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Cree : {md}")
    print(f"Cree : {js}")


if __name__ == "__main__":
    main()
