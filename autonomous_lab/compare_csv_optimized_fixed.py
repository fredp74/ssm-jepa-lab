import itertools
import json
import random
from pathlib import Path

from acquire_csv_skills import CHECKS, make_cases
from compare_csv_exploration import read, write
from csv_executable_skills import execute


def main():
    root = Path("autonomous_lab")
    old = read(root / "runs/csv_ambiguity_diagnostic_01/results.json")
    comparison = read(root / "runs/csv_exploration_comparison_01/results.json")
    programs = read(root / "data/csv_skill_acquisition_01/programs.json")
    catalog = read(root / "data/csv_ssm_policy_01/catalog.json")
    output = root / "runs/csv_optimized_fixed_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    by_id = {p["program_id"]: p for p in programs}
    tests = {t["test_id"]: t for t in catalog}
    environments = old["results"]
    development_cases = make_cases(catalog)

    responses = {
        p["program_id"]: {
            i: execute(p["program"], t["input"])
            for i, t in tests.items()
        }
        for p in programs
    }
    development_predictions = {
        p["program_id"]: [
            execute(p["program"], rows) for rows in development_cases
        ]
        for p in programs
    }
    coverage_cache = {}

    def coverage(ids):
        key = tuple(sorted(ids))
        if key not in coverage_cache:
            coverage_cache[key] = sum(
                len({
                    development_predictions[i][k] for i in key
                }) == 1
                for k in range(len(development_cases))
            )
        return coverage_cache[key]

    def apply_pair(environment, pair):
        observations = environment["observations"]
        ids = list(environment["surviving_program_ids"])
        calls = len(observations)
        cells = sum(
            len(row) for o in observations for row in o["input"]
        )
        size = sum(
            len(v.encode("utf-8"))
            for o in observations for row in o["input"] for v in row
        )
        actions = []
        for i in pair:
            test = tests[i]
            already_used = any(
                test["input"] == o["input"] for o in observations
            )
            if (
                already_used or calls >= 8
                or cells + test["cells"] > 24
                or size + test["input_text_utf8_bytes"] > 512
            ):
                continue
            answer = responses[environment["environment_program_id"]][i]
            ids = [p for p in ids if responses[p][i] == answer]
            actions.append(i)
            calls += 1
            cells += test["cells"]
            size += test["input_text_utf8_bytes"]
        return ids, actions, cells, size

    # Optimisation exclusivement sur les anciens tableaux.
    best = None
    for pair in itertools.permutations(sorted(tests), 2):
        answered = calls = cells = size = 0
        for environment in environments:
            ids, actions, used_cells, used_bytes = apply_pair(environment, pair)
            answered += coverage(ids)
            calls += len(actions)
            cells += used_cells
            size += used_bytes
        # Couverture d'abord, puis cout et ordre deterministe.
        score = (-answered, calls, cells, size, pair)
        if best is None or score < best[0]:
            best = (score, pair)

    pair = best[1]
    output.mkdir(parents=True)
    write(output / "protocol.json", {
        "development_environments_reused": True,
        "fixed_pair": list(pair),
        "pair_selected_using_old_100_tables_only": True,
        "development_consensus_answers": -best[0][0],
        "searched_ordered_pairs": len(tests) * (len(tests) - 1),
        "maximum_additional_calls": 2,
        "total_budget": {"calls": 8, "cells": 24, "text_bytes": 512},
        "fresh_evaluation_seed": 118001,
        "fresh_evaluation_tables": 200,
        "scope": "new_tables_in_known_program_family",
    })
    print(f"Paire fixe figee : {pair}")

    # L'adaptative est reutilisee sans changer ses decisions.
    adaptive = {
        r["environment_program_id"]: r
        for r in comparison["results"]
        if r["strategy"] == "adaptive"
    }
    frozen = []
    for environment in environments:
        eid = environment["environment_program_id"]
        ids, actions, cells, size = apply_pair(environment, pair)
        frozen.append({
            "strategy": "optimized_fixed",
            "environment_program_id": eid,
            "surviving_program_ids": ids,
            "selected_test_ids": actions,
            "cells": cells,
            "text_bytes": size,
        })
        run = adaptive[eid]
        frozen.append({
            "strategy": "adaptive",
            "environment_program_id": eid,
            "surviving_program_ids": run["surviving_program_ids"],
            "selected_test_ids": run["selected_test_ids"],
            "cells": run["cells"],
            "text_bytes": run["text_bytes"],
        })
    write(output / "frozen_runs.json", frozen)

    forbidden = (
        [t["input"] for t in catalog] + CHECKS + development_cases
        + read(root / "runs/csv_exploration_comparison_01/evaluation_cases.json")
    )
    values = [
        "", "00092", "0084", "7.35", "-5.25", "TRUE", "yes", "FALSE",
        "e\u0301_holdout", "UPPER_holdout", "  edge_holdout  ",
        "inner   gap_holdout", "w" * 31, 'quote"holdout',
        "comma,holdout", "line\nholdout", "東京_holdout",
    ]
    rng = random.Random(118001)
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

    predictions = {
        p["program_id"]: [
            execute(p["program"], rows) for rows in cases
        ]
        for p in programs
    }
    summaries = {}
    for run in frozen:
        eid = run["environment_program_id"]
        ids = run["surviving_program_ids"]
        if eid not in ids:
            raise RuntimeError("Programme reel elimine.")
        answered = correct = 0
        for k, expected in enumerate(predictions[eid]):
            outputs = {predictions[i][k] for i in ids}
            if len(outputs) == 1:
                answered += 1
                correct += int(next(iter(outputs)) == expected)
        run.update(answered=answered, correct=correct)
        group = summaries.setdefault(run["strategy"], {
            "strategy": run["strategy"],
            "answered": 0, "correct": 0, "evaluation_cases": 0,
            "additional_calls": 0, "total_cells": 0, "total_text_bytes": 0,
        })
        group["answered"] += answered
        group["correct"] += correct
        group["evaluation_cases"] += len(cases)
        group["additional_calls"] += len(run["selected_test_ids"])
        group["total_cells"] += run["cells"]
        group["total_text_bytes"] += run["text_bytes"]

    summary = list(summaries.values())
    for group in summary:
        group["coverage"] = group["answered"] / group["evaluation_cases"]
        group["errors"] = group["answered"] - group["correct"]
        print(json.dumps(group))
    write(output / "results.json", {
        "protocol": read(output / "protocol.json"),
        "summary": summary,
        "results": frozen,
    })
    print(f"Resultats : {output / 'results.json'}")


if __name__ == "__main__":
    main()
