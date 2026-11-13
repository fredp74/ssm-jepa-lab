import json
import random
from collections import Counter
from pathlib import Path

from csv_environment import configurations, render
from csv_executable_skills import execute, save_skill, load_skill

CHECKS = [
    [["verify_B"], ["verify_A"]],
    [["duplicate_V"], ["duplicate_V"]],
    [[""], ["kept_V"]],
    [["00053", "e\u0301_V"]],
]

def make_cases(catalog):
    forbidden = [test["input"] for test in catalog] + CHECKS
    values = [
        "", "00067", "0091", "3.75", "-2.50", "TRUE", "yes",
        "e\u0301_skill", "UPPER_skill", "  edge_skill  ",
        "inner   gap_skill", "q"*27, 'quote"skill',
        "comma,skill", "line\nskill", "東京_skill",
    ]
    rng = random.Random(116001)
    cases = []
    while len(cases) < 100:
        height, width = rng.randint(1, 4), rng.randint(1, 4)
        rows = [
            [rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if rows not in cases and rows not in forbidden:
            cases.append(rows)
    return cases

def main():
    root = Path("autonomous_lab")
    data = root / "data/csv_skill_acquisition_01"
    programs = json.loads(
        (data / "programs.json").read_text(encoding="utf-8")
    )
    classes = list(json.loads(
        (data / "observable_classes.json").read_text(encoding="utf-8")
    ).values())
    catalog = json.loads(
        (root / "data/csv_ssm_policy_01/catalog.json").read_text(
            encoding="utf-8"
        )
    )
    output = root / "runs/csv_skill_acquisition_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    representatives = [
        programs[ids[0]]["program"] for ids in classes
    ]
    responses = [
        [execute(program, test["input"]) for test in catalog]
        for program in representatives
    ]
    cases = make_cases(catalog)
    assert not any(
        check == test["input"] for check in CHECKS for test in catalog
    )

    # Le simulateur couvre quatre configurations par transformation.
    environments = [
        item for item in programs
        if item["configuration_id"] in (0, 7, 23, 47)
    ]

    output.mkdir(parents=True)
    (output / "evaluation_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = {
        "maximum_calls": 8,
        "maximum_cells": 24,
        "maximum_text_bytes": 512,
        "maximum_discovery_calls": 4,
        "discovery_cell_budget": 16,
        "discovery_text_byte_budget": 384,
        "verification_calls_required": 4,
        "proposal_frozen_before_verification": True,
        "status_after_checks": "provisional",
        "hypothesis_language": "programmed_individual_transformations_plus_csv",
        "reserved_compositions_used": False,
        "neural_training": False,
        "environments": len(environments),
        "seed": 116001,
    }
    (output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    results = []
    for number, environment in enumerate(environments):
        hidden_program = environment["program"]
        observations = []
        used = []
        calls = cells = text_bytes = 0
        candidates = list(range(len(classes)))

        def query(rows):
            nonlocal calls, cells, text_bytes
            calls += 1
            cells += sum(len(row) for row in rows)
            text_bytes += sum(
                len(value.encode("utf-8")) for row in rows for value in row
            )
            assert calls <= 8 and cells <= 24 and text_bytes <= 512
            answer = execute(hidden_program, rows)
            observations.append({
                "input": [list(row) for row in rows],
                "output": answer,
            })
            return answer

        while len(candidates) > 1 and len(used) < 4:
            available = [
                test["test_id"] for test in catalog
                if test["test_id"] not in used
                and cells + test["cells"] <= 16
                and text_bytes + test["input_text_utf8_bytes"] <= 384
            ]
            if not available:
                break

            def score(index):
                groups = Counter(
                    responses[candidate][index] for candidate in candidates
                )
                return (
                    sum(size*size for size in groups.values()),
                    catalog[index]["cells"],
                    catalog[index]["input_text_utf8_bytes"],
                    index,
                )

            selected = min(available, key=score)
            answer = query(catalog[selected]["input"])
            used.append(selected)
            candidates = [
                candidate for candidate in candidates
                if responses[candidate][selected] == answer
            ]
            if not candidates:
                break

        status = "ambiguous" if candidates else "outside_candidate_family"
        predicted_program = None
        checks = []

        if len(candidates) == 1:
            predicted_program = representatives[candidates[0]]
            status = "provisional"
            for rows in CHECKS:
                prediction = execute(predicted_program, rows)
                answer = query(rows)
                checks.append({
                    "input": rows,
                    "predicted": prediction,
                    "output": answer,
                })
                if prediction != answer:
                    status = "revoked"
                    break

        # Reference limitee aux formats CSV initiaux, avec les memes retours.
        configs = configurations()
        base_id = max(
            range(len(configs)),
            key=lambda i: sum(
                render(item["input"], configs[i]) == item["output"]
                for item in observations
            ),
        )
        expected = [execute(hidden_program, rows) for rows in cases]
        baseline_exact = sum(
            render(rows, configs[base_id]) == answer
            for rows, answer in zip(cases, expected)
        )

        acquired_exact = 0
        acquired_answered = 0
        if status == "provisional":
            path = output / f"skill_{number:03d}.json"
            discovery_observations = observations[:len(used)]
            save_skill(path, predicted_program, discovery_observations, checks)
            restored = load_skill(path)
            calls_before = calls
            predictions = [
                execute(restored["program"], rows) for rows in cases
            ]
            assert calls == calls_before
            acquired_answered = len(predictions)
            acquired_exact = sum(
                prediction == answer
                for prediction, answer in zip(predictions, expected)
            )

        result = {
            "environment_program_id": environment["program_id"],
            "operations": hidden_program["operations"],
            "configuration": environment["configuration_id"],
            "status": status,
            "remaining_observable_classes": len(candidates),
            "calls": calls,
            "cells": cells,
            "text_bytes": text_bytes,
            "baseline_exact": baseline_exact,
            "acquired_exact": acquired_exact,
            "acquired_answered": acquired_answered,
            "evaluation_cases": len(cases),
            "selected_test_ids": used,
        }
        results.append(result)

    summary = {
        "environments": len(results),
        "statuses": dict(Counter(row["status"] for row in results)),
        "baseline_exact": sum(row["baseline_exact"] for row in results),
        "acquired_exact": sum(row["acquired_exact"] for row in results),
        "acquired_answered": sum(row["acquired_answered"] for row in results),
        "evaluation_cases": sum(row["evaluation_cases"] for row in results),
        "fully_correct_acquired_environments": sum(
            row["status"] == "provisional"
            and row["acquired_exact"] == row["evaluation_cases"]
            for row in results
        ),
        "mean_calls": sum(row["calls"] for row in results)/len(results),
    }
    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": config, "summary": summary, "results": results,
        }, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Competences et resultats : {output}")

if __name__ == "__main__":
    main()
