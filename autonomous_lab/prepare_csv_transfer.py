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
    output = root / "data/csv_transfer_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    original_path = root / "data/csv_composition_01/protocol.json"
    original = read(original_path)
    operations = original["primitive_operations"]
    development_pairs = {
        tuple(e["operations"]) for e in original["environments"]
    }

    rng = random.Random(122001)
    available_pairs = [
        pair for pair in itertools.permutations(operations, 2)
        if pair not in development_pairs
    ]
    pairs = rng.sample(sorted(available_pairs), 12)
    triples = rng.sample(
        sorted(itertools.permutations(operations, 3)), 6
    )
    configs = (0, 7, 23, 47)
    environments = []

    def add(group, sequence, extra_operation=None):
        for configuration_id in configs:
            environments.append({
                "environment_id": len(environments),
                "group": group,
                "configuration_id": configuration_id,
                "operations": list(sequence),
                "extra_operation_after_sequence": extra_operation,
            })

    add("simple_control", [])
    add("simple_control", ["casefold"])
    for pair in pairs:
        add("heldout_development_pair", pair)
    for triple in triples:
        add("three_operations", triple)
    add("unrepresented_primitive", [], "reverse_columns")

    # Exclure les tableaux deja disponibles localement.
    catalog_path = root / "data/csv_ssm_policy_01/catalog.json"
    forbidden = [t["input"] for t in read(catalog_path)]
    previous_case_paths = sorted(
        (root / "runs").glob("*/evaluation_cases.json")
    )
    for path in previous_case_paths:
        forbidden.extend(read(path))

    learning_inputs = [
        [["A", "00042"], ["a", "00042"]],
        [["  ", ""], ["e\u0301_verify ", "7.25"]],
    ]
    verification_inputs = [
        [
            ["z_verify"], ["a_verify"],
            ["abcdefghijklmnop_X"], ["abcdefghijklmnop_Y"],
        ],
        [["e\u0301_check "], ["é_check "], ["   "], ["00048"]],
    ]
    forbidden += learning_inputs + verification_inputs

    values = [
        "", "   ", "00076", "0082", "9.25", "-6.75",
        "TRUE", "yes", "FALSE", "A", "a",
        "e\u0301_transfer", "é_transfer", "UPPER_transfer",
        "  edge_transfer  ", "inner   gap_transfer",
        "collisionprefix__X", "collisionprefix__Y",
        'quote"transfer', "comma,transfer",
        "line\ntransfer", "東京_transfer",
    ]
    case_rng = random.Random(122002)
    cases = []
    while len(cases) < 200:
        height = case_rng.randint(1, 4)
        width = case_rng.randint(1, 4)
        rows = [
            [case_rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if height >= 2 and case_rng.random() < 0.25:
            rows[-1] = list(rows[0])
        if rows not in forbidden and rows not in cases:
            cases.append(rows)

    dependencies = [
        root / "acquire_csv_compositions_v3.py",
        root / "csv_environment.py",
        root / "csv_policy_behaviors.py",
        catalog_path,
        original_path,
    ]
    dependencies += previous_case_paths

    fingerprints = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dependencies
    }
    protocol = {
        "experiment": "frozen_composition_transfer",
        "preparation_only": True,
        "evaluation_executed": False,
        "agent_version": "acquire_csv_compositions_v3",
        "agent_and_dependencies_sha256": fingerprints,
        "environment_seed": 122001,
        "evaluation_seed": 122002,
        "evaluation_tables": len(cases),
        "configuration_ids": list(configs),
        "environment_count": len(environments),
        "two_operation_pairs": [list(pair) for pair in pairs],
        "excluded_development_pairs": [
            list(pair) for pair in sorted(development_pairs)
        ],
        "three_operation_sequences": [
            list(triple) for triple in triples
        ],
        "two_operation_pairs_absent_from_six_development_cases": True,
        "absence_from_all_historical_experiments_verified": False,
        "extra_operation_definition": (
            "reverse_columns reverses each row before CSV rendering; "
            "it is not included in the agent candidate language"
        ),
        "out_of_family_label_requires_behavioral_witness": True,
        "three_operations_may_be_equivalent_to_shorter_programs": True,
        "agent_receives_environment_group_or_program": False,
        "learning_inputs": learning_inputs,
        "verification_inputs": verification_inputs,
        "budget": {
            "calls": 8, "cells": 24, "input_text_utf8_bytes": 512
        },
        "allocation": {
            "catalogue": {"calls": 4, "cells": 8, "bytes": 256},
            "learning": {"calls": 2, "cells": 8, "bytes": 128},
            "verification": {"calls": 2, "cells": 8, "bytes": 128},
        },
        "required_metrics": [
            "coverage_and_errors_by_group",
            "acceptance_rejection_and_abstention",
            "tool_and_computation_cost",
            "prediction_persistence",
            "behavioral_distinguishability_from_candidate_family",
        ],
        "evaluation_feedback_used_for_selection": False,
        "neural_training": False,
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
    print("Controles simples : 8")
    print("Paires absentes des six cas de developpement : 48")
    print("Sequences de trois operations : 24")
    print("Inversion des colonnes : 4")
    print(f"Tableaux reserves : {len(cases)}")
    print("Agent et dependances : empreintes enregistrees")
    print("Aucune evaluation executee.")
    print(f"Protocole : {output / 'protocol.json'}")


if __name__ == "__main__":
    main()
