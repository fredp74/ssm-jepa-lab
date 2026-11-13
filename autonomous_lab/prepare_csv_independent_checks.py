import hashlib
import itertools
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
    output = root / "data/csv_independent_checks_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    grammar_path = root / "data/csv_composition_01/protocol.json"
    previous_path = root / "data/csv_transfer_01/environments.json"
    grammar = read(grammar_path)
    previous = read(previous_path)
    operations = grammar["primitive_operations"]

    excluded_pairs = {
        tuple(e["operations"])
        for e in previous + grammar["environments"]
        if len(e["operations"]) == 2
    }
    excluded_triples = {
        tuple(e["operations"])
        for e in previous
        if len(e["operations"]) == 3
    }
    rng = random.Random(126001)
    pairs = rng.sample([
        p for p in sorted(itertools.permutations(operations, 2))
        if p not in excluded_pairs
    ], 12)
    triples = rng.sample([
        p for p in sorted(itertools.permutations(operations, 3))
        if p not in excluded_triples
    ], 8)

    environments = []

    def add(group, sequence, extra=None):
        for configuration_id in (0, 7, 23, 47):
            environments.append({
                "environment_id": len(environments),
                "group": group,
                "configuration_id": configuration_id,
                "operations": list(sequence),
                "extra_operation_after_sequence": extra,
            })

    add("simple_control", [])
    add("simple_control", ["casefold"])
    for pair in pairs:
        add("reserved_pair", pair)
    for triple in triples:
        add("reserved_three_operations", triple)
    add("unrepresented_primitive", [], "rotate_columns_left")

    # Exclure les tableaux deja prepares ou utilises.
    forbidden = []
    prior_paths = []
    for directory in (root / "data", root / "runs"):
        for path in sorted(directory.glob("*/evaluation_cases.json")):
            forbidden.extend(read(path))
            prior_paths.append(path)

    probes_path = root / "data/csv_counterexamples_01/probes.json"
    forbidden += [p["input"] for p in read(probes_path)]

    values = [
        "", " ", "   ", "A", "a", "B", "z",
        "00054", "0054", "54", "3.25", "-7.50",
        "TRUE", "true", "yes", "FALSE", "no",
        "e\u0301", "é", " x ", "x", "a  b", "a b",
        "abcdefghijklmnop_X", "abcdefghijklmnop_Y",
        'q"r', "a,b", "a;b", "a\tb", "a\nb", "東京",
    ]

    # Dix paires de contre-tests, chacune de huit cellules au total.
    check_banks = []
    all_checks = []
    for seed in range(126101, 126111):
        check_rng = random.Random(seed)
        checks = []
        while len(checks) < 2:
            height, width = check_rng.choice([(4, 1), (2, 2), (1, 4)])
            rows = [
                [check_rng.choice(values) for _ in range(width)]
                for _ in range(height)
            ]
            if height > 1 and check_rng.random() < 0.25:
                rows[-1] = list(rows[0])
            used_bytes = sum(
                len(v.encode("utf-8"))
                for table in checks + [rows]
                for row in table for v in row
            )
            if (
                used_bytes <= 128
                and rows not in forbidden
                and rows not in all_checks
                and rows not in checks
            ):
                checks.append(rows)
        check_banks.append({"seed": seed, "inputs": checks})
        all_checks.extend(checks)

    evaluation_rng = random.Random(126002)
    cases = []
    while len(cases) < 200:
        height = evaluation_rng.randint(1, 4)
        width = evaluation_rng.randint(1, 4)
        rows = [
            [evaluation_rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if height > 1 and evaluation_rng.random() < 0.25:
            rows[-1] = list(rows[0])
        if (
            rows not in forbidden
            and rows not in all_checks
            and rows not in cases
        ):
            cases.append(rows)

    dependencies = [
        grammar_path, previous_path, probes_path,
        root / "evaluate_csv_generated_transfer.py",
        root / "csv_environment.py",
        root / "csv_policy_behaviors.py",
        Path(__file__),
    ]
    protocol = {
        "experiment": "independent_verification_inputs",
        "preparation_only": True,
        "evaluation_executed": False,
        "environment_seed": 126001,
        "evaluation_seed": 126002,
        "environment_count": len(environments),
        "evaluation_tables": len(cases),
        "fixed_discovery_sequence": [463, 399, 0],
        "learning_inputs": "unchanged_from_v3",
        "arms": [
            "original_fixed_verification",
            "independent_checks_10_seeds",
        ],
        "independent_checks_selected_using_tool_feedback": False,
        "independent_checks_selected_using_surviving_hypotheses": False,
        "check_values_informed_by_known_primitives": True,
        "verification_calls_per_environment": 2,
        "verification_cells_per_environment": 8,
        "verification_maximum_text_bytes": 128,
        "total_budget": {
            "calls": 8, "cells": 24, "input_text_utf8_bytes": 512
        },
        "candidate_family": "unchanged_up_to_two_operations",
        "reserved_sequences_excluded_from": [
            str(grammar_path), str(previous_path)
        ],
        "absence_from_all_historical_experiments_verified": False,
        "extra_operation_definition": (
            "rotate_columns_left moves the first cell of each row "
            "to the end before CSV rendering"
        ),
        "out_of_family_status_requires_behavioral_witness": True,
        "three_operations_may_match_shorter_programs": True,
        "primary_metrics": [
            "accepted_environments_with_evaluation_errors",
            "errors_among_answered_queries",
            "correct_answers_lost_to_abstention",
        ],
        "secondary_metrics": [
            "coverage_by_group",
            "actual_calls_cells_bytes",
            "offline_computation_time",
            "persistence",
            "variation_across_check_seeds",
        ],
        "evaluation_feedback_used_for_selection": False,
        "ten_check_seeds_are_not_ten_independent_behavior_benchmarks": True,
        "neural_training": False,
        "dependencies_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dependencies
        },
    }

    output.mkdir(parents=True)
    write(output / "environments.json", environments)
    write(output / "check_banks.json", check_banks)
    write(output / "evaluation_cases.json", cases)
    protocol["prepared_files_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in (
            "environments.json", "check_banks.json",
            "evaluation_cases.json",
        )
    }
    write(output / "protocol.json", protocol)

    print(f"Environnements : {len(environments)}")
    print("Controles : 8 | paires : 48 | triples : 32 | rotation : 4")
    print(f"Banques de verification independantes : {len(check_banks)}")
    print(f"Tableaux reserves : {len(cases)}")
    print("Apprentissage fixe ; deux appels de verification par bras.")
    print("Empreintes enregistrees ; aucune evaluation executee.")
    print(f"Protocole : {output / 'protocol.json'}")


if __name__ == "__main__":
    main()
