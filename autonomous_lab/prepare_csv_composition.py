import hashlib
import itertools
import json
from pathlib import Path

from csv_environment import configurations
from csv_policy_behaviors import simulate


def main():
    root = Path("autonomous_lab")
    source = root / "data/csv_skill_acquisition_01/programs.json"
    output = root / "data/csv_composition_01"

    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    previous = json.loads(source.read_text(encoding="utf-8-sig"))
    operations = sorted({
        operation
        for item in previous
        for operation in item["program"]["operations"]
    })
    if len(operations) != 12:
        raise SystemExit(
            f"12 operations attendues, {len(operations)} trouvees."
        )

    sequences = [[]]
    sequences += [[operation] for operation in operations]
    sequences += [
        list(pair) for pair in itertools.permutations(operations, 2)
    ]
    configs = configurations()

    # Ces compositions sont un banc de developpement.
    # Aucune nouveaute par rapport aux experiences precedentes n'est supposee.
    compositions = [
        ["casefold", "deduplicate_rows"],
        ["rstrip", "remove_empty_rows"],
        ["truncate_16", "deduplicate_rows"],
        ["normalize_leading_zeros", "sort_rows"],
        ["unicode_NFC", "rstrip"],
        ["decimal_comma", "sort_rows"],
    ]
    for sequence in compositions:
        if sequence not in sequences:
            raise RuntimeError(f"Composition absente : {sequence}")

    environments = [
        {
            "environment_id": number * 4 + position,
            "configuration_id": configuration_id,
            "operations": sequence,
        }
        for number, sequence in enumerate(compositions)
        for position, configuration_id in enumerate((0, 7, 23, 47))
    ]

    # Controle de plomberie : l'ordre des operations a bien un effet.
    rows = [["A"], ["a"]]
    first = simulate(
        rows, configs[0], ["casefold", "deduplicate_rows"]
    )
    reverse = simulate(
        rows, configs[0], ["deduplicate_rows", "casefold"]
    )
    if first == reverse:
        raise RuntimeError("Le controle de l'ordre a echoue.")

    protocol = {
        "experiment": "symbolic_composition_acquisition",
        "development_benchmark": True,
        "previously_explored_compositions_may_be_present": True,
        "claim_of_unseen_behavior_generalization": False,
        "primitive_operations": operations,
        "candidate_operation_sequences": sequences,
        "configuration_count": len(configs),
        "candidate_program_count": len(configs) * len(sequences),
        "maximum_operations_per_program": 2,
        "repeated_identical_operations_included": False,
        "candidate_generation": "exhaustive_enumeration",
        "grammar_written_by_humans": True,
        "neural_training": False,
        "environments": environments,
        "budget": {
            "maximum_calls": 8,
            "maximum_cells": 24,
            "maximum_input_text_utf8_bytes": 512,
            "maximum_discovery_calls": 6,
            "discovery_cells": 16,
            "discovery_input_text_utf8_bytes": 384,
            "verification_calls_reserved": 2,
            "verification_cells_reserved": 8,
            "verification_input_text_utf8_bytes_reserved": 128,
        },
        "acceptance": {
            "proposal_frozen_before_verification": True,
            "require_all_survivors_to_agree_on_verification": True,
            "verification_failure": "revoke",
            "success_status": "provisional",
            "independent_evaluation_feedback_used_for_selection": False,
        },
        "comparisons": [
            "single_operation_candidate_family",
            "up_to_two_operations_candidate_family",
        ],
        "required_measurements": [
            "coverage",
            "accuracy_among_answered_queries",
            "verification_failures",
            "remaining_candidates",
            "tool_calls_cells_and_bytes",
            "offline_search_time",
            "prediction_after_save_and_reload",
        ],
        "source_programs_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "order_sensitivity_plumbing_check_passed": True,
        "agent_acquisition_tested": False,
    }

    output.mkdir(parents=True)
    path = output / "protocol.json"
    path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Operations elementaires : {len(operations)}")
    print(f"Sequences candidates : {len(sequences)}")
    print(f"Programmes candidats : {protocol['candidate_program_count']}")
    print(f"Environnements : {len(environments)}")
    print("Ordre des operations : controle reussi")
    print("Aucune acquisition ni evaluation executee.")
    print(f"Protocole : {path}")


if __name__ == "__main__":
    main()
