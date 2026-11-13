import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    runs = Path("autonomous_lab/runs")
    sources = {
        "simple_controls": runs / "csv_counterexamples_01/results.json",
        "fixed_vs_adaptive": runs / "csv_generated_transfer_01/results.json",
    }
    docs = Path("docs")
    md = docs / "csv-generated-observations.md"
    js = docs / "csv-generated-observations.json"
    if md.exists() or js.exists():
        raise SystemExit("Documentation existante : aucun remplacement.")

    data = {name: read(path) for name, path in sources.items()}
    controls = data["simple_controls"]
    transfer = data["fixed_vs_adaptive"]

    def find(rows, arm):
        return next(row for row in rows if row["arm"] == arm)

    baseline = find(controls["summary"], "original_catalogue")
    generated = find(controls["summary"], "generated_inputs")
    adaptive = find(transfer["summary"], "generated_adaptive")
    fixed = find(transfer["summary"], "generated_fixed")

    table = "\n".join(
        f"| {row['arm']} | {row['group']} | "
        f"{row['answered']:,}/{row['evaluation_cases']:,} | "
        f"{row['errors']} | {row['coverage']:.2%} |"
        for row in transfer["group_summary"]
    )

    paired = {}
    for row in transfer["results"]:
        paired.setdefault(row["environment_id"], {})[
            row["arm"]
        ] = row
    equal_coverage = sum(
        rows["generated_adaptive"]["answered"]
        == rows["generated_fixed"]["answered"]
        for rows in paired.values()
    )
    persistence = {
        name: {
            "checked": len(result["results"]),
            "passed": sum(
                row["memory_roundtrip_passed"]
                for row in result["results"]
            ),
        }
        for name, result in data.items()
    }

    text = f"""# Generated CSV observations: fixed versus adaptive selection

A bank of automatically generated inputs substantially improved
coverage within the predefined candidate family. A fixed sequence
selected from the simple-control experiment matched adaptive selection
on all 84 environments, with much less online search computation.

Errors on three-operation environments remained unresolved.

## Generated input bank

The bank contains 512 inputs constructed from generic templates and
bounded random sampling. Values are informed by the known primitive
operations. Hidden environment programs are not used to generate inputs.

This is selection from a generated finite bank, not unrestricted input
synthesis or invention of new primitive operations.

Both arms use the same expanded candidate family: up to two ordered
operations combined with CSV configurations.

Selection minimizes expected remaining response classes under a uniform
prior over original-catalogue response classes. Learning inputs and
verification inputs remain unchanged from v3.

## Simple-control comparison

Eight environments cover identity and casefold across four CSV
configurations.

| Observation bank | Correct answers | Coverage | Calls | Cells |
|---|---:|---:|---:|---:|
| Original catalogue | {baseline['correct']:,}/{baseline['evaluation_cases']:,} | {baseline['coverage']:.1%} | {baseline['total_calls']} | {baseline['total_cells']} |
| Generated inputs | {generated['correct']:,}/{generated['evaluation_cases']:,} | {generated['coverage']:.1%} | {generated['total_calls']} | {generated['total_cells']} |

Both arms had zero observed errors among answered questions.
The generated bank used more calls, the same number of cells and fewer
input-text bytes.

All eight generated-bank runs selected tests 463, 399 and 0.
That observation motivated a fixed-sequence reference.

Passing verification did not imply unique program identification.
Some accepted memories retained multiple hypotheses and still abstained
on evaluation questions.

## Fixed versus adaptive comparison

The sequence **463, 399, 0** was frozen before evaluation on new tables
across the 84 previously examined environments.

These environments were reused development evidence, including the
earlier transfer failures. This is not an independent evaluation of
previously unexamined behavior families.

Both arms share the generated bank, candidate family, learning inputs,
verification inputs and resource ceilings:

- Eight total calls.
- Twenty-four cells.
- Five hundred twelve input-text UTF-8 bytes.

Discovery receives up to four calls and eight cells. The fixed arm
follows its three-test sequence, skipping infeasible tests without
substituting other tests.

| Strategy | Group | Answers | Errors | Coverage |
|---|---|---:|---:|---:|
{table}

Each strategy answered {fixed['answered']:,} of
{fixed['evaluation_cases']:,} questions, with {fixed['correct']:,}
correct answers and {fixed['errors']} errors.

Coverage was equal in {equal_coverage}/{len(paired)} paired environments.
The aggregate error counts were also equal. This does not establish
that all individual predictions or retained hypotheses were identical.

Both strategies used {fixed['total_calls']} calls.
Cell consumption was {adaptive['total_cells']} for adaptive selection
and {fixed['total_cells']} for the fixed sequence. Input-text consumption
was {adaptive['total_text_bytes']:,} versus
{fixed['total_text_bytes']:,} bytes.

## Computation

In the simple-control experiment, generated-bank response precomputation
took {controls['generated_bank_precomputation_seconds']:.2f} seconds,
versus {controls['shared_catalogue_precomputation_seconds']:.2f} seconds
for the original catalogue.

In the 84-environment comparison, generated-bank precomputation took
{transfer['generated_bank_precomputation_seconds']:.2f} seconds and was
shared by both arms.

Cumulative search and verification time was
{adaptive['search_and_verification_seconds']:.2f} seconds for the
adaptive arm and {fixed['search_and_verification_seconds']:.2f} seconds
for the fixed arm.

These are descriptive timings from one run, not replicated benchmarks
or complete end-to-end deployment costs.

## Persistence

Payload and prediction roundtrip checks passed for
{persistence['simple_controls']['passed']}/
{persistence['simple_controls']['checked']} memories in the controls
experiment and {persistence['fixed_vs_adaptive']['passed']}/
{persistence['fixed_vs_adaptive']['checked']} in the larger comparison.

The checks compare predictions on catalogue and verification inputs
before saving and after reloading. Evaluation uses reloaded programs.

## Interpretation

The generated bank offers more useful observations under the cell
budget. No benefit from adaptive selection was measured over the
selected fixed sequence in this experiment.

The candidate family represents the tested two-operation compositions,
but not every three-operation behavior. The latter group produced
135 errors per arm despite passing checks in some environments.

Agreement among retained candidates cannot guarantee correctness when
the true behavior is outside the candidate family. Column reversal was
detected as incompatible, but that success does not establish general
unknown-behavior detection.

The expanded repertoire, generated bank and explicit memory remain
human-designed symbolic mechanisms. There is no neural training and no
demonstrated SSM/JEPA advantage.

## Limitations

The fixed sequence was selected using the simple-control results.
The larger comparison was frozen before execution, but the environments
had already been examined in earlier experiments.

Fresh evaluation tables do not make these behavior families unseen.
Tables are shared across environments, so question counts are not
independent experimental replications.

The comparison with the earlier original-catalogue transfer experiment
uses different evaluation tables. Its percentages are not a paired
question-level comparison.

Any subsequent method change must treat these results as consulted
development evidence.

## Reproduction

From the repository root, with prior composition artifacts available:

1. python autonomous_lab/prepare_csv_counterexamples.py
2. python autonomous_lab/make_csv_counterexample_evaluator.py
3. python autonomous_lab/evaluate_csv_counterexamples.py
4. python autonomous_lab/prepare_csv_generated_transfer.py
5. python autonomous_lab/make_csv_generated_transfer_evaluator.py
6. python autonomous_lab/evaluate_csv_generated_transfer.py
7. python autonomous_lab/summarize_csv_generated_observations.py

Generators and experiments refuse to overwrite existing destinations.
Prepared protocols fingerprint dependencies and input files.
Raw runs remain local under autonomous_lab/runs.
"""

    report = {
        "conclusion": (
            "Generated observations improved in-family coverage. "
            "The selected fixed sequence matched adaptive coverage "
            "with less online search computation. "
            "Out-of-family errors remained."
        ),
        "paired_environments_with_equal_coverage": equal_coverage,
        "persistence": persistence,
        "sources": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "protocol": data[name]["protocol"],
                "summary": data[name]["summary"],
                "group_summary": data[name].get("group_summary"),
                "generated_bank_precomputation_seconds":
                    data[name]["generated_bank_precomputation_seconds"],
            }
            for name, path in sources.items()
        },
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
