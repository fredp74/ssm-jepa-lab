import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from acquire_csv_skills import CHECKS, make_cases
from csv_executable_skills import execute


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    root = Path("autonomous_lab")
    source = root / "runs/csv_ambiguity_diagnostic_01/results.json"
    program_path = root / "data/csv_skill_acquisition_01/programs.json"
    catalog_path = root / "data/csv_ssm_policy_01/catalog.json"
    output = root / "runs/csv_exploration_comparison_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    previous = read(source)
    programs = read(program_path)
    catalog = read(catalog_path)
    by_id = {p["program_id"]: p for p in programs}
    tests = {t["test_id"]: t for t in catalog}
    fixed_order = [11] + sorted(i for i in tests if i != 11)
    seeds = list(range(117101, 117121))

    protocol = {
        "development_environments_reused": True,
        "evaluation": "new_tables_in_known_program_family",
        "additional_calls_per_environment": 2,
        "total_budget": {"calls": 8, "cells": 24, "text_bytes": 512},
        "fixed_order": fixed_order,
        "fixed_order_informed_by_previous_diagnostic": True,
        "random_seeds": seeds,
        "evaluation_seed": 117001,
        "evaluation_tables": 200,
        "evaluation_feedback_used_for_selection": False,
        "all_program_aliases_checked": True,
        "neural_training": False,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source, program_path, catalog_path, Path(__file__))
        },
    }
    output.mkdir(parents=True)
    write(output / "protocol.json", protocol)

    rng = random.Random(117001)
    forbidden = (
        [t["input"] for t in catalog] + CHECKS + make_cases(catalog)
    )
    values = [
        "", "00081", "0073", "6.25", "-4.75", "TRUE", "yes", "FALSE",
        "e\u0301_fresh", "UPPER_fresh", "  edge_fresh  ",
        "inner   gap_fresh", "z" * 29, 'quote"fresh',
        "comma,fresh", "line\nfresh", "東京_fresh",
    ]
    cases = []
    while len(cases) < 200:
        height, width = rng.randint(1, 4), rng.randint(1, 4)
        rows = [
            [rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if rows not in forbidden and rows not in cases:
            cases.append(rows)
    write(output / "evaluation_cases.json", cases)

    # Cache du catalogue uniquement, sans les tableaux d'evaluation.
    responses = {
        p["program_id"]: {
            i: execute(p["program"], t["input"])
            for i, t in tests.items()
        }
        for p in programs
    }

    runs = []
    arms = [("adaptive", None), ("fixed", None)]
    arms += [("random", seed) for seed in seeds]

    for old in previous["results"]:
        environment = old["environment_program_id"]
        observations = old["observations"]

        for strategy, seed in arms:
            survivors = list(old["surviving_program_ids"])
            calls = len(observations)
            cells = sum(
                len(row) for o in observations for row in o["input"]
            )
            text_bytes = sum(
                len(v.encode("utf-8"))
                for o in observations for row in o["input"] for v in row
            )
            used = {
                i for i, t in tests.items()
                if any(t["input"] == o["input"] for o in observations)
            }
            local_rng = random.Random(f"{seed}:{environment}")
            actions = []

            for _ in range(2):
                feasible = [
                    i for i, t in tests.items()
                    if i not in used and calls < 8
                    and cells + t["cells"] <= 24
                    and text_bytes + t["input_text_utf8_bytes"] <= 512
                ]
                if not feasible:
                    break

                if strategy == "fixed":
                    selected = next(i for i in fixed_order if i in feasible)
                elif strategy == "random":
                    selected = local_rng.choice(sorted(feasible))
                else:
                    representatives = {}
                    for i in survivors:
                        representatives.setdefault(
                            by_id[i]["catalog_response_signature"], i
                        )

                    def score(i):
                        groups = Counter(
                            responses[p][i]
                            for p in representatives.values()
                        )
                        return (
                            sum(n*n for n in groups.values()),
                            tests[i]["cells"],
                            tests[i]["input_text_utf8_bytes"],
                            i,
                        )

                    selected = min(feasible, key=score)

                answer = responses[environment][selected]
                survivors = [
                    i for i in survivors
                    if responses[i][selected] == answer
                ]
                if environment not in survivors:
                    raise RuntimeError("Programme reel elimine.")

                actions.append(selected)
                used.add(selected)
                calls += 1
                cells += tests[selected]["cells"]
                text_bytes += tests[selected]["input_text_utf8_bytes"]

            runs.append({
                "environment_program_id": environment,
                "strategy": strategy,
                "seed": seed,
                "selected_test_ids": actions,
                "surviving_program_ids": survivors,
                "calls": calls,
                "cells": cells,
                "text_bytes": text_bytes,
            })

    # Tous les choix sont sauvegardes avant de mesurer les predictions.
    write(output / "frozen_runs.json", runs)

    predictions = {
        p["program_id"]: [
            execute(p["program"], rows) for rows in cases
        ]
        for p in programs
    }
    summaries = {}
    for run in runs:
        expected = predictions[run["environment_program_id"]]
        answered = correct = 0
        for k, truth in enumerate(expected):
            outputs = {
                predictions[i][k] for i in run["surviving_program_ids"]
            }
            if len(outputs) == 1:
                answered += 1
                correct += int(next(iter(outputs)) == truth)

        run.update(
            answered=answered, correct=correct, evaluation_cases=len(cases)
        )
        key = (run["strategy"], run["seed"])
        group = summaries.setdefault(key, {
            "strategy": key[0], "seed": key[1],
            "environments": 0, "answered": 0, "correct": 0,
            "evaluation_cases": 0, "additional_calls": 0,
        })
        group["environments"] += 1
        group["answered"] += answered
        group["correct"] += correct
        group["evaluation_cases"] += len(cases)
        group["additional_calls"] += len(run["selected_test_ids"])

    summary = list(summaries.values())
    for group in summary:
        group["coverage"] = group["answered"] / group["evaluation_cases"]
        group["errors"] = group["answered"] - group["correct"]
        print(json.dumps(group))

    write(output / "results.json", {
        "protocol": protocol, "summary": summary, "results": runs,
    })
    print(f"Resultats : {output / 'results.json'}")


if __name__ == "__main__":
    main()
