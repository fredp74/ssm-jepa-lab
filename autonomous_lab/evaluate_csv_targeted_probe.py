import json
from collections import Counter
from pathlib import Path

from acquire_csv_skills import make_cases
from csv_executable_skills import execute


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def measure(programs, cases, expected):
    answered = correct = 0
    for rows, truth in zip(cases, expected):
        outputs = {execute(program, rows) for program in programs}
        if len(outputs) == 1:
            answered += 1
            correct += int(next(iter(outputs)) == truth)
    return {"answered": answered, "correct": correct}


def main():
    root = Path("autonomous_lab")
    source = root / "runs/csv_ambiguity_diagnostic_01/results.json"
    previous = read_json(source)
    programs = read_json(
        root / "data/csv_skill_acquisition_01/programs.json"
    )
    catalog = read_json(
        root / "data/csv_ssm_policy_01/catalog.json"
    )
    output = root / "runs/csv_targeted_probe_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    by_id = {item["program_id"]: item for item in programs}
    cases = make_cases(catalog)
    results = []

    for old in previous["results"]:
        environment_id = old["environment_program_id"]
        hidden = by_id[environment_id]["program"]
        observations = old["observations"]
        survivors = [
            by_id[program_id]
            for program_id in old["surviving_program_ids"]
        ]

        calls = len(observations)
        cells = sum(
            len(row)
            for observation in observations
            for row in observation["input"]
        )
        text_bytes = sum(
            len(value.encode("utf-8"))
            for observation in observations
            for row in observation["input"]
            for value in row
        )

        feasible = [
            test for test in catalog
            if calls + 1 <= 8
            and cells + test["cells"] <= 24
            and text_bytes + test["input_text_utf8_bytes"] <= 512
            and not any(
                test["input"] == observation["input"]
                for observation in observations
            )
        ]

        # Les alias du catalogue recoivent ensemble le poids d'une classe.
        groups = {}
        for item in survivors:
            groups.setdefault(
                item["catalog_response_signature"], []
            ).append(item)
        representatives = [items[0] for items in groups.values()]

        def score(test):
            counts = Counter(
                execute(item["program"], test["input"])
                for item in representatives
            )
            return (
                sum(size * size for size in counts.values()),
                test["cells"],
                test["input_text_utf8_bytes"],
                test["test_id"],
            )

        selected = None
        after = survivors
        observation = None
        if feasible:
            candidate = min(feasible, key=score)
            # Aucun appel si le meilleur test ne departage aucune classe.
            if score(candidate)[0] < len(representatives) ** 2:
                selected = candidate
                answer = execute(hidden, selected["input"])
                observation = {
                    "input": selected["input"],
                    "output": answer,
                }
                after = [
                    item for item in survivors
                    if execute(item["program"], selected["input"]) == answer
                ]
                calls += 1
                cells += selected["cells"]
                text_bytes += selected["input_text_utf8_bytes"]

        if environment_id not in {
            item["program_id"] for item in after
        }:
            raise RuntimeError("Le programme reel a ete elimine.")
        if calls > 8 or cells > 24 or text_bytes > 512:
            raise RuntimeError("Budget depasse.")

        # Evaluation seulement apres le choix du test et le filtrage.
        expected = [execute(hidden, rows) for rows in cases]
        before_score = measure(
            [item["program"] for item in survivors], cases, expected
        )
        after_score = measure(
            [item["program"] for item in after], cases, expected
        )
        if before_score != {
            "answered": old["answered"],
            "correct": old["correct"],
        }:
            raise RuntimeError("Le resultat precedent n'est pas reproduit.")

        result = {
            "environment_program_id": environment_id,
            "operations": old["operations"],
            "configuration": old["configuration"],
            "selected_test_id": (
                selected["test_id"] if selected else None
            ),
            "additional_observation": observation,
            "programs_before": len(survivors),
            "programs_after": len(after),
            "classes_after": len({
                item["catalog_response_signature"] for item in after
            }),
            "surviving_program_ids": [
                item["program_id"] for item in after
            ],
            "calls": calls,
            "cells": cells,
            "text_bytes": text_bytes,
            "before": before_score,
            "after": after_score,
            "evaluation_cases": len(cases),
        }
        results.append(result)
        print(
            f"Programme {environment_id} | "
            f"test={result['selected_test_id']} | "
            f"hypotheses={len(survivors)}->{len(after)} | "
            f"reponses={before_score['answered']}"
            f"->{after_score['answered']}/{len(cases)}"
        )

    before = sum(row["before"]["answered"] for row in results)
    after = sum(row["after"]["answered"] for row in results)
    correct = sum(row["after"]["correct"] for row in results)
    extra_calls = sum(
        row["selected_test_id"] is not None for row in results
    )
    total = sum(row["evaluation_cases"] for row in results)
    summary = {
        "environments": len(results),
        "evaluation_cases": total,
        "additional_calls": extra_calls,
        "answered_before": before,
        "answered_after": after,
        "additional_answers": after - before,
        "correct_after": correct,
        "errors_after": after - correct,
        "coverage_before": before / total if total else None,
        "coverage_after": after / total if total else None,
        "additional_answers_per_call": (
            (after - before) / extra_calls if extra_calls else None
        ),
    }
    output.mkdir(parents=True)
    (output / "results.json").write_text(
        json.dumps({
            "protocol": {
                "retrospective_diagnostic": True,
                "evaluation_cases_reused": True,
                "maximum_additional_calls_per_environment": 1,
                "total_budget": {
                    "calls": 8, "cells": 24, "text_bytes": 512
                },
                "selection": "minimum_expected_remaining_catalog_classes",
                "class_prior": "uniform",
                "evaluation_feedback_used_for_selection": False,
                "all_surviving_program_aliases_checked": True,
                "reserved_compositions_used": False,
                "neural_training": False,
            },
            "summary": summary,
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Resultats : {output / 'results.json'}")


if __name__ == "__main__":
    main()
