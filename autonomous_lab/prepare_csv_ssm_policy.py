import hashlib
import json
from pathlib import Path

root = Path("autonomous_lab/data/csv_ssm_policy_01")
if root.exists():
    raise SystemExit("Protocole deja present : aucun remplacement.")

sources = [
    Path("autonomous_lab/data/csv_diversity_01/probe_banks.json"),
    Path("autonomous_lab/data/csv_contrasts_01/probe_banks.json"),
    Path("autonomous_lab/data/csv_common_budget_01/plans.json"),
]

catalog = []
seen = {}
source_hashes = {}

def add(rows, provenance):
    encoded = json.dumps(
        rows, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()

    if key in seen:
        catalog[seen[key]]["provenance"].append(provenance)
        return

    cells = sum(len(row) for row in rows)
    text_bytes = sum(
        len(value.encode("utf-8")) for row in rows for value in row
    )
    assert cells <= 24 and text_bytes <= 512

    seen[key] = len(catalog)
    catalog.append({
        "test_id": len(catalog),
        "input": rows,
        "cells": cells,
        "input_text_utf8_bytes": text_bytes,
        "sha256": key,
        "provenance": [provenance],
    })

for source in sources:
    raw = source.read_bytes()
    source_hashes[str(source)] = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))

    for name, bank in data.get("banks", {}).items():
        for index, rows in enumerate(bank):
            add(rows, f"{source}:{name}:{index}")

    for name, plan in data.get("plans", {}).items():
        for index, call in enumerate(plan):
            add(call["input"], f"{source}:{name}:{index}")

train_operations = [
    "normalize_leading_zeros",
    "unicode_NFC",
    "casefold",
    "rstrip",
    "collapse_spaces",
    "decimal_comma",
    "boolean_normalization",
    "truncate_16",
    "sort_rows",
    "remove_empty_rows",
    "deduplicate_rows",
    "drop_empty_columns",
]

protocol = {
    "goal": "choose_predefined_tests_for_black_box_csv_behavior",
    "stage": "protocol_and_catalog_only",
    "maximum_calls": 8,
    "maximum_cells": 24,
    "maximum_input_text_utf8_bytes": 512,
    "byte_budget_excludes_transport_encoding": True,
    "training_behaviors": {
        "identity_included": True,
        "individual_operations": train_operations,
    },
    "reserved_behavior_compositions": [
        ["unicode_NFC", "rstrip"],
        ["decimal_comma", "sort_rows"],
        ["normalize_leading_zeros", "drop_empty_columns"],
        ["casefold", "deduplicate_rows"],
    ],
    "composition_order": "listed_order_before_csv_serialization",
    "evaluation_scope": "new_compositions_of_familiar_operations",
    "not_claimed": [
        "invention_of_tests",
        "discovery_of_new_primitives",
        "learning_without_prior_training",
    ],
    "student": "SSM_test_selection_policy",
    "training_reference": "symbolic_teacher_on_training_behaviors_only",
    "verification": "same_exact_known_family_consistency_checker",
    "evaluation_parameter_updates": 0,
    "baseline_policies": [
        "fixed_feasible_order",
        "random_feasible_selection",
        "symbolic_training_family_policy",
    ],
    "required_pretraining_audit": [
        "which_behaviors_are_detectable_by_the_catalog",
        "detection_possible_within_all_three_budgets",
        "whether_distinct_training_behaviors_have_identical_observations",
    ],
    "primary_metrics": [
        "unknown_behavior_detection",
        "false_rejection_of_known_configurations",
        "actual_calls_cells_and_bytes",
        "wall_clock_time_including_policy_inference",
    ],
    "reserved_feedback_must_not_train_teacher_or_student": True,
    "SSM_and_baselines_must_share_catalog_and_budgets": True,
    "JEPA_not_in_first_comparison": True,
    "catalog_sources_sha256": source_hashes,
    "catalog_size": len(catalog),
}

root.mkdir(parents=True)
(root / "catalog.json").write_text(
    json.dumps(catalog, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(root / "protocol.json").write_text(
    json.dumps(protocol, indent=2), encoding="utf-8",
)

print(f"Catalogue fige : {len(catalog)} tests distincts")
print("Budget : 8 appels / 24 cellules / 512 octets")
print("Apprentissage : transformations individuelles")
print("Evaluation reservee : quatre compositions")
print(f"Fichiers : {root}")
