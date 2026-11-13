import hashlib
import json
import statistics
from pathlib import Path


def main():
    source = Path("autonomous_lab/runs/csv_independent_checks_01/results.json")
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    docs = Path("docs")
    md = docs / "csv-independent-verification.md"
    js = docs / "csv-independent-verification.json"
    if md.exists() or js.exists():
        raise SystemExit("Documentation existante : aucun remplacement.")

    overall = [r for r in data["summary"] if r["group"] == "ALL"]
    fixed = next(r for r in overall if r["arm"] == "fixed")
    independent = [r for r in overall if r["arm"] != "fixed"]

    def stats(key):
        values = [r[key] for r in independent]
        return {
            "mean": statistics.mean(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    metrics = {
        key: stats(key)
        for key in (
            "coverage", "answered", "correct", "errors",
            "accepted_environments_with_errors",
            "fixed_acceptances_with_errors_rejected",
            "fixed_correct_answers_lost_by_environment",
        )
    }
    trial_table = "\n".join(
        f"| {r['arm']} | {r['coverage']:.2%} | "
        f"{r['errors']} | {r['accepted_environments_with_errors']} |"
        for r in independent
    )
    persistence_passed = sum(
        r["prediction_roundtrip_passed"] for r in data["results"]
    )

    text = f"""# Independent CSV verification inputs

Two independently sampled verification inputs did not provide a robust
improvement over the existing fixed checks.

The average number of incorrect answers decreased, but coverage also
decreased. The average number of accepted environments with evaluation
errors increased. Outcomes varied substantially across check seeds.

## Protocol

The benchmark contains 92 environments:

- Eight simple controls.
- Forty-eight environments using reserved two-operation sequences.
- Thirty-two environments using reserved three-operation sequences.
- Four environments that rotate columns left.

Reserved sequences were excluded from the specified earlier development
manifests. Absence from the entire historical experiment record was not
established.

Discovery uses the fixed sequence 463, 399, 0. Learning observations,
candidate programs and resource ceilings are shared across all arms.

The reference uses the two v3 verification inputs. Ten alternative arms
each use two independently sampled inputs, selected without tool feedback
or surviving-hypothesis information. Their value pool is informed by the
known primitives.

Hypotheses are frozen before verification. Verification responses do not
filter the hypotheses. They determine whether the frozen memory is
provisionally accepted, revoked or left ambiguous.

All choices are saved before evaluation on 200 prepared tables per
environment. The ten check seeds share the same environments and tables;
they are not ten independent behavior benchmarks.

## Overall results

| Metric | Fixed verification | Independent checks, mean |
|---|---:|---:|
| Answer coverage | {fixed['coverage']:.2%} | {metrics['coverage']['mean']:.2%} |
| Answers provided | {fixed['answered']:,} | {metrics['answered']['mean']:,.1f} |
| Correct answers | {fixed['correct']:,} | {metrics['correct']['mean']:,.1f} |
| Incorrect answers | {fixed['errors']} | {metrics['errors']['mean']:.1f} |
| Accepted environments with errors | {fixed['accepted_environments_with_errors']} | {metrics['accepted_environments_with_errors']['mean']:.1f} |

Each arm has {fixed['evaluation_cases']:,} evaluation questions.

Independent-check errors ranged from {metrics['errors']['minimum']} to
{metrics['errors']['maximum']}. Coverage ranged from
{metrics['coverage']['minimum']:.2%} to
{metrics['coverage']['maximum']:.2%}.

A lower absolute error count cannot by itself establish better
verification when the procedure also answers fewer questions.

## All check seeds

| Arm | Coverage | Incorrect answers | Accepted environments with errors |
|---|---:|---:|---:|
{trial_table}

No seed is selected as a winner after inspecting these results.

## Results by behavior group

All observed errors were in the three-operation group.

The fixed reference answered 92.88% of questions in the reserved-pair
group, with zero observed errors. Independent checks had lower average
coverage in that group, also with zero observed errors.

The rotation operation was rejected in all four configurations by every
arm. This does not establish detection of arbitrary unknown behavior.

Independent checks rejected some erroneous fixed-reference acceptances,
but could also accept environments on which the reference abstained.
Counting only corrected reference failures would miss those new
acceptances.

The reported fixed_correct_answers_lost_by_environment metric sums
positive per-environment decreases in correct-answer counts. It is not
a question-level count of matching predictions lost.

## Resources and persistence

Every arm used {fixed['total_calls']} calls and
{fixed['total_cells']} cells. Actual input-byte consumption can differ.

Learning observations were computed once per environment and shared
between verification arms. These are simulated logical tool-call
comparisons, not repeated deployments against an external service.

Shared generated-probe response precomputation took
{data['shared_precomputation_seconds']:.2f} seconds. Shared learning took
{data['shared_learning_seconds']:.2f} seconds on the reported machine.
These timings are descriptive, not replicated benchmarks.

Payload and prediction roundtrip checks passed for
{persistence_passed}/{len(data['results'])} memories. Evaluation used
candidate identities validated against the reloaded memory payloads.

## Interpretation and limitations

Independent sampling alone does not make two checks sufficiently
informative. Some draws reduce errors through increased abstention;
others allow more erroneous acceptances.

Candidate agreement remains conditional on the candidate language.
A true behavior outside that language can agree with retained programs
on observed inputs and disagree later.

Three operations do not automatically imply an unrepresentable behavior.
The evaluator separately records whether any expanded-family program
matches the full evaluation response vector. That finite-table
diagnostic is not a proof of global equivalence.

The environments and value pools are synthetic. The same evaluation
tables are shared across related environments, so question counts are
not independent replications.

There is no neural training and no demonstrated SSM/JEPA advantage.
This experiment does not establish robust autonomous learning or a
general verification guarantee.

These results are now consulted development evidence for future changes.

## Reproduction

After the generated-observation experiments:

1. python autonomous_lab/prepare_csv_independent_checks.py
2. python autonomous_lab/evaluate_csv_independent_checks.py
3. python autonomous_lab/summarize_csv_independent_checks.py

Preparation records dependency and input hashes. Evaluation checks them
before running. Existing destinations are not overwritten.

Raw runs remain local under autonomous_lab/runs/csv_independent_checks_01.
"""

    report = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "protocol": data["protocol"],
        "summary": data["summary"],
        "independent_check_statistics": metrics,
        "persistence": {
            "checked": len(data["results"]),
            "passed": persistence_passed,
        },
        "conclusion": (
            "Two independently sampled checks did not robustly improve "
            "verification. Lower average error counts coincided with "
            "lower coverage and more accepted environments with errors."
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
