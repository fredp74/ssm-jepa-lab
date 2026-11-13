import hashlib
import json
import statistics
from pathlib import Path


def main():
    root = Path("autonomous_lab/runs")
    sources = {
        "ambiguity": root / "csv_ambiguity_diagnostic_01/results.json",
        "targeted": root / "csv_targeted_probe_01/results.json",
        "comparison": root / "csv_exploration_comparison_01/results.json",
        "optimized": root / "csv_optimized_fixed_01/results.json",
    }
    docs = Path("docs")
    md = docs / "csv-consensus-and-exploration.md"
    js = docs / "csv-consensus-and-exploration.json"
    if md.exists() or js.exists():
        raise SystemExit("Documentation existante : aucun remplacement.")

    data = {
        name: json.loads(path.read_text(encoding="utf-8-sig"))
        for name, path in sources.items()
    }

    ambiguity = data["ambiguity"]["summary"]
    targeted = data["targeted"]["summary"]
    comparison = data["comparison"]["summary"]
    optimized = data["optimized"]

    def find(rows, strategy):
        return next(r for r in rows if r["strategy"] == strategy)

    first_adaptive = find(comparison, "adaptive")
    first_fixed = find(comparison, "fixed")
    random_rows = [r for r in comparison if r["strategy"] == "random"]
    fixed = find(optimized["summary"], "optimized_fixed")
    adaptive = find(optimized["summary"], "adaptive")
    pair = optimized["protocol"]["fixed_pair"]

    paired = {}
    for row in optimized["results"]:
        paired.setdefault(row["environment_program_id"], {})[
            row["strategy"]
        ] = row["answered"]

    wins = {"fixed": 0, "adaptive": 0, "ties": 0}
    for row in paired.values():
        delta = row["optimized_fixed"] - row["adaptive"]
        key = "fixed" if delta > 0 else "adaptive" if delta < 0 else "ties"
        wins[key] += 1

    random_mean = statistics.mean(r["coverage"] for r in random_rows)
    random_min = min(r["coverage"] for r in random_rows)
    random_max = max(r["coverage"] for r in random_rows)

    text = f"""# CSV consensus predictions and exploration controls

The optimized fixed pair outperformed the current adaptive heuristic.
An advantage of adaptation has not been established.

## Scope

These experiments revisit the 26 environments left ambiguous by the
executable-skill acquisition pilot. Candidate programs combine predefined
individual transformations with CSV configurations.

A question is answered only when every surviving program produces the
same output. Otherwise, the system abstains. All program aliases are
checked individually.

This is symbolic inference within a programmed hypothesis language.
No neural training is performed. These results do not demonstrate
open-ended skill invention or an SSM/JEPA learning advantage.

## Retrospective diagnostics

On the original evaluation questions, consensus answered
{ambiguity['consensus_answered']:,}/{ambiguity['evaluation_cases']:,}
correctly, with {ambiguity['coverage']:.1%} coverage.

One additional targeted call per environment increased coverage to
{targeted['coverage_after']:.1%}, yielding
{targeted['answered_after']:,} correct answers and zero observed errors.
All environments selected test 11.

Both diagnostics reused the original evaluation tables. Choosing the
same test everywhere did not establish a benefit from adaptation.

## Initial exploration comparison

Each strategy received up to two additional calls per environment.
The total resource ceiling was eight calls, 24 cells and 512 input-text
UTF-8 bytes, including the shared initial observations.

A new set of 200 tables was evaluated in each of the 26 environments.

| Strategy | Coverage |
|---|---:|
| Adaptive heuristic | {first_adaptive['coverage']:.1%} |
| Simple fixed sequence | {first_fixed['coverage']:.1%} |
| Random, mean across {len(random_rows)} seeds | {random_mean:.1%} |

Random coverage ranged from {random_min:.1%} to {random_max:.1%}.
Each strategy used 52 additional calls per run. There were zero observed
errors among answered queries.

The fixed sequence started with test 11, informed by the previous
diagnostic, and then followed catalogue order. It was not optimized.

The adaptive heuristic minimizes the expected number of remaining
catalogue-response classes under a uniform class prior.

## Optimized fixed-pair control

All {optimized['protocol']['searched_ordered_pairs']:,} ordered pairs
were searched using the original 100 development tables and reused
environments. Selection maximized consensus coverage, with resource cost
and deterministic order breaking ties.

The chosen pair, **{pair[0]} then {pair[1]}**, was saved before evaluation
on another 200 new tables per environment. Adaptive decisions were
reused unchanged.

| Strategy | Correct answers | Coverage | Additional calls |
|---|---:|---:|---:|
| Optimized fixed pair | {fixed['correct']:,}/{fixed['evaluation_cases']:,} | {fixed['coverage']:.1%} | {fixed['additional_calls']} |
| Adaptive heuristic | {adaptive['correct']:,}/{adaptive['evaluation_cases']:,} | {adaptive['coverage']:.1%} | {adaptive['additional_calls']} |

The fixed pair gained {fixed['answered'] - adaptive['answered']:,}
answers, or {(fixed['coverage'] - adaptive['coverage']) * 100:.1f}
percentage points. Both strategies had zero observed errors.

The fixed pair had higher coverage in {wins['fixed']} environments;
the adaptive heuristic had higher coverage in {wins['adaptive']};
{wins['ties']} were tied.

Total cell consumption was {fixed['total_cells']} for the fixed pair
versus {adaptive['total_cells']} for the adaptive heuristic.
Total input-text consumption was {fixed['total_text_bytes']:,}
versus {adaptive['total_text_bytes']:,} bytes.

Call counts were equal and resource ceilings were shared, but actual
cell and byte consumption differed.

## Interpretation and limitations

The optimized fixed pair is the stronger reference in this experiment.

Reducing candidate-class uncertainty and maximizing prediction coverage
are different objectives. The fixed-pair search optimized development
coverage directly; the adaptive heuristic optimized class uncertainty.
This comparison does not rule out better adaptive policies.

The fixed pair was optimized offline using simulated development
environments and their known programs. That search cost is not included
in the reported tool-call budget.

The environments and transformation language were reused. New tables
measure transfer to new inputs within that family, not discovery of
unseen transformation families or arbitrary tools.

The true simulated program remains among the candidates. Under that
assumption, unanimous predictions must agree with it. Zero observed
errors do not establish reliability outside the candidate family.

The 5,200 questions reuse 200 tables across 26 related environments.
They are not 5,200 independent experimental replications.

## Reproduction

Run the following scripts from the repository root, after the
executable-skill acquisition pilot:

1. python autonomous_lab/diagnose_csv_ambiguity.py
2. python autonomous_lab/evaluate_csv_targeted_probe.py
3. python autonomous_lab/compare_csv_exploration.py
4. python autonomous_lab/compare_csv_optimized_fixed.py
5. python autonomous_lab/summarize_csv_consensus.py

Experiment scripts refuse to overwrite existing output directories.
Raw results remain local under autonomous_lab/runs.
The companion JSON records source summaries and SHA-256 hashes.
"""

    summary = {
        "conclusion": (
            "The optimized fixed pair outperformed the current adaptive "
            "heuristic. An advantage of adaptation was not established."
        ),
        "paired_environment_comparison": wins,
        "sources": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "protocol": data[name].get("protocol"),
                "summary": data[name]["summary"],
            }
            for name, path in sources.items()
        },
    }

    docs.mkdir(exist_ok=True)
    md.write_text(text.rstrip() + "\n", encoding="utf-8")
    js.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Cree : {md}")
    print(f"Cree : {js}")


if __name__ == "__main__":
    main()
