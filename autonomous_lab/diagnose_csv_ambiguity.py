import json
from pathlib import Path

from acquire_csv_skills import make_cases
from csv_executable_skills import execute


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    root = Path("autonomous_lab")
    data = root / "data/csv_skill_acquisition_01"
    previous = read_json(
        root / "runs/csv_skill_acquisition_01/results.json"
    )
    programs = read_json(data / "programs.json")
    catalog = read_json(
        root / "data/csv_ssm_policy_01/catalog.json"
    )
    output = root / "runs/csv_ambiguity_diagnostic_01"

    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    by_id = {item["program_id"]: item for item in programs}
    tests = {item["test_id"]: item for item in catalog}
    cases = make_cases(catalog)
    results = []
    details = []

    for old in previous["results"]:
        if old["status"] != "ambiguous":
            continue

        environment_id = old["environment_program_id"]
        hidden = by_id[environment_id]["program"]

        # Reconstitution des observations du pilote.
        # Aucun retour des questions d'evaluation ne filtre les candidats.
        observations = [
            {
                "input": tests[test_id]["input"],
                "output": execute(hidden, tests[test_id]["input"]),
            }
            for test_id in old["selected_test_ids"]
        ]

        # Tous les programmes sont conserves individuellement :
        # aucune equivalence globale n'est supposee entre les alias.
        survivors = [
            item for item in programs
            if all(
                execute(item["program"], observation["input"])
                == observation["output"]
                for observation in observations
            )
        ]
        if environment_id not in {
            item["program_id"] for item in survivors
        }:
            raise RuntimeError("Le programme reel a ete elimine.")

        # Les hypotheses sont figees avant toute evaluation.
        frozen = tuple(item["program"] for item in survivors)
        answered = correct = 0

        for case_id, rows in enumerate(cases):
            predictions = {
                execute(program, rows) for program in frozen
            }
            unanimous = len(predictions) == 1
            prediction = next(iter(predictions)) if unanimous else None

            # Le retour reel sert uniquement a mesurer le resultat.
            expected = execute(hidden, rows)
            answered += int(unanimous)
            correct += int(unanimous and prediction == expected)
            details.append({
                "environment_program_id": environment_id,
                "case_id": case_id,
                "distinct_predictions": len(predictions),
                "answered": unanimous,
                "prediction": prediction,
                "expected": expected,
            })

        result = {
            "environment_program_id": environment_id,
            "operations": old["operations"],
            "configuration": old["configuration"],
            "remaining_programs": len(survivors),
            "remaining_catalog_classes": len({
                item["catalog_response_signature"]
                for item in survivors
            }),
            "surviving_program_ids": [
                item["program_id"] for item in survivors
            ],
            "observations": observations,
            "answered": answered,
            "correct": correct,
            "evaluation_cases": len(cases),
        }
        results.append(result)
        print(
            f"Programme {environment_id} | "
            f"hypotheses={len(survivors)} | "
            f"reponses={answered}/{len(cases)} | "
            f"correctes={correct}/{answered}"
        )

    total = sum(row["evaluation_cases"] for row in results)
    answered = sum(row["answered"] for row in results)
    correct = sum(row["correct"] for row in results)
    summary = {
        "ambiguous_environments": len(results),
        "evaluation_cases": total,
        "consensus_answered": answered,
        "consensus_correct": correct,
        "consensus_errors": answered - correct,
        "abstentions": total - answered,
        "coverage": answered / total if total else None,
        "accuracy_when_answered": correct / answered if answered else None,
    }

    output.mkdir(parents=True)
    (output / "results.json").write_text(
        json.dumps({
            "protocol": {
                "retrospective_diagnostic": True,
                "evaluation_cases_reused": True,
                "all_program_aliases_checked": True,
                "evaluation_feedback_used_for_selection": False,
                "new_tool_calls": 0,
                "scope": "programmed_candidate_family",
                "neural_training": False,
            },
            "summary": summary,
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "prediction_details.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in details
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Resultats : {output / 'results.json'}")


if __name__ == "__main__":
    main()
