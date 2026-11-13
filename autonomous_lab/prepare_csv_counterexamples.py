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
    output = root / "data/csv_counterexamples_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    composition_path = root / "data/csv_composition_01/protocol.json"
    catalog_path = root / "data/csv_ssm_policy_01/catalog.json"
    composition = read(composition_path)

    # Valeurs inspirees des primitives connues, sans consulter
    # les programmes reels des environnements.
    values = [
        "", " ", "  ", "A", "a", "B",
        "0007", "07", "7", "2.50", "-1.25",
        "TRUE", "true", "yes", "no",
        "e\u0301", "é", " x ", "x",
        "a  b", "a b", "abcdefghijklmnop_X",
        "abcdefghijklmnop_Y", 'q"r', "a,b", "a;b",
        "a\tb", "a\nb", "東京",
    ]
    rng = random.Random(123001)
    probes = []
    seen = set()

    def add(rows):
        key = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        if key in seen:
            return
        seen.add(key)
        cells = sum(len(row) for row in rows)
        text_bytes = sum(
            len(value.encode("utf-8"))
            for row in rows for value in row
        )
        probes.append({
            "test_id": len(probes),
            "input": rows,
            "cells": cells,
            "input_text_utf8_bytes": text_bytes,
            "sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
        })

    # Formes generiques : cellule seule, repetition, contraste.
    for value in values:
        add([[value]])
        add([[value], [value]])

    while len(probes) < 512:
        height, width = rng.choice([(2, 1), (1, 2), (2, 2), (3, 1)])
        rows = [
            [rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if height > 1 and rng.random() < 0.25:
            rows[-1] = list(rows[0])
        add(rows)

    environments = [
        {
            "environment_id": number * 4 + position,
            "configuration_id": configuration_id,
            "operations": operations,
        }
        for number, operations in enumerate(([], ["casefold"]))
        for position, configuration_id in enumerate((0, 7, 23, 47))
    ]

    dependencies = [
        composition_path,
        catalog_path,
        root / "csv_environment.py",
        root / "csv_policy_behaviors.py",
        root / "acquire_csv_compositions_v3.py",
        Path(__file__),
    ]
    protocol = {
        "experiment": "generated_counterexample_inputs",
        "stage": "simple_controls_development",
        "preparation_only": True,
        "evaluation_executed": False,
        "seed": 123001,
        "probe_count": len(probes),
        "probe_generation": "bounded_random_and_generic_templates",
        "probe_values_informed_by_known_primitives": True,
        "true_environment_program_used_for_probe_generation": False,
        "free_form_input_synthesis": False,
        "candidate_operation_sequences":
            composition["candidate_operation_sequences"],
        "environments": environments,
        "budget": {
            "calls": 8, "cells": 24, "input_text_utf8_bytes": 512
        },
        "arms": [
            "original_catalogue",
            "generated_inputs",
        ],
        "shared_selection_objective":
            "minimize_expected_remaining_response_classes",
        "candidate_prior": "uniform_catalogue_response_classes",
        "shared_learning_and_verification": "v3",
        "allocation": {
            "discovery": {"calls": 4, "cells": 8, "bytes": 256},
            "learning": {"calls": 2, "cells": 8, "bytes": 128},
            "verification": {"calls": 2, "cells": 8, "bytes": 128},
        },
        "required_metrics": [
            "coverage",
            "errors_among_answered_queries",
            "provisional_environments",
            "remaining_candidates",
            "actual_calls_cells_and_bytes",
            "offline_search_time",
            "prediction_persistence",
        ],
        "evaluation_feedback_used_for_selection": False,
        "neural_training": False,
        "dependencies_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dependencies
        },
    }

    output.mkdir(parents=True)
    write(output / "probes.json", probes)
    protocol["probes_sha256"] = hashlib.sha256(
        (output / "probes.json").read_bytes()
    ).hexdigest()
    write(output / "protocol.json", protocol)

    print(f"Entrees generees : {len(probes)}")
    print(f"Controles simples : {len(environments)}")
    print("Comparaison : catalogue initial / entrees generees")
    print("Objectif de selection et budgets communs")
    print("Aucune interrogation ni evaluation executee.")
    print(f"Protocole : {output / 'protocol.json'}")


if __name__ == "__main__":
    main()
