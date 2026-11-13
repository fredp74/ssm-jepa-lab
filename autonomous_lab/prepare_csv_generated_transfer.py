import hashlib
import json
import random
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    root = Path("autonomous_lab")
    output = root / "data/csv_generated_transfer_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    probes_path = root / "data/csv_counterexamples_01/probes.json"
    environments_path = root / "data/csv_transfer_01/environments.json"
    previous_path = root / "runs/csv_counterexamples_01/results.json"
    probes = read(probes_path)
    environments = read(environments_path)
    previous = read(previous_path)

    generated_runs = [
        row for row in previous["results"]
        if row["arm"] == "generated_inputs"
    ]
    paths = {
        tuple(row["selected_test_ids"]) for row in generated_runs
    }
    if paths != {(463, 399, 0)}:
        raise SystemExit(f"Sequence precedente inattendue : {paths}")

    forbidden = [p["input"] for p in probes]
    previous_cases = sorted(
        (root / "runs").glob("*/evaluation_cases.json")
    )
    for path in previous_cases:
        forbidden.extend(read(path))

    values = [
        "", "   ", "00093", "0086", "4.75", "-8.25",
        "TRUE", "true", "yes", "FALSE", "A", "a",
        "e\u0301_comparison", "é_comparison",
        "UPPER_comparison", "  edge_comparison  ",
        "inner   gap_comparison",
        "collisionprefix__X", "collisionprefix__Y",
        'quote"comparison', "comma,comparison",
        "line\ncomparison", "東京_comparison",
    ]
    rng = random.Random(125001)
    cases = []
    while len(cases) < 200:
        height, width = rng.randint(1, 4), rng.randint(1, 4)
        rows = [
            [rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if height >= 2 and rng.random() < 0.25:
            rows[-1] = list(rows[0])
        if rows not in forbidden and rows not in cases:
            cases.append(rows)

    dependencies = [
        probes_path,
        environments_path,
        previous_path,
        root / "evaluate_csv_counterexamples.py",
        root / "data/csv_composition_01/protocol.json",
        root / "data/csv_ssm_policy_01/catalog.json",
        root / "csv_environment.py",
        root / "csv_policy_behaviors.py",
        Path(__file__),
    ]

    protocol = {
        "experiment": "generated_inputs_fixed_vs_adaptive",
        "preparation_only": True,
        "evaluation_executed": False,
        "development_environments_reused": True,
        "earlier_transfer_results_already_consulted": True,
        "independent_unseen_behavior_evaluation": False,
        "environment_count": len(environments),
        "evaluation_seed": 125001,
        "evaluation_tables": len(cases),
        "candidate_family": "same_up_to_two_operations_in_both_arms",
        "probe_bank": "same_512_generated_inputs_in_both_arms",
        "arms": ["generated_adaptive", "generated_fixed"],
        "fixed_sequence": [463, 399, 0],
        "fixed_sequence_selected_from_previous_simple_controls": True,
        "fixed_action_rule": (
            "Follow this order; skip an infeasible test; "
            "do not replace it with another test."
        ),
        "adaptive_selection":
            "minimize_expected_remaining_response_classes",
        "candidate_prior": "uniform_original_catalogue_response_classes",
        "learning_and_verification": "unchanged_from_counterexamples_01",
        "budget": {
            "calls": 8, "cells": 24, "input_text_utf8_bytes": 512
        },
        "allocation": {
            "discovery": {"calls": 4, "cells": 8, "bytes": 256},
            "learning": {"calls": 2, "cells": 8, "bytes": 128},
            "verification": {"calls": 2, "cells": 8, "bytes": 128},
        },
        "evaluation_feedback_used_for_selection": False,
        "required_metrics": [
            "coverage_and_errors_by_group",
            "paired_environment_comparison",
            "acceptance_and_abstention",
            "calls_cells_bytes_and_computation_time",
            "prediction_persistence",
        ],
        "dependencies_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dependencies
        },
    }

    output.mkdir(parents=True)
    write(output / "environments.json", environments)
    write(output / "evaluation_cases.json", cases)
    protocol["prepared_files_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("environments.json", "evaluation_cases.json")
    }
    write(output / "protocol.json", protocol)

    print(f"Environnements : {len(environments)}")
    print("Strategies : sequence fixe / selection adaptative")
    print("Sequence fixe : 463 -> 399 -> 0")
    print(f"Nouveaux tableaux : {len(cases)}")
    print("Budgets et repertoire communs ; empreintes enregistrees.")
    print("Aucune evaluation executee.")
    print(f"Protocole : {output / 'protocol.json'}")


if __name__ == "__main__":
    main()
