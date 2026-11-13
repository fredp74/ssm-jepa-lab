import hashlib
import itertools
import json
import random
from pathlib import Path

def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    root = Path("autonomous_lab")
    output = root / "data/csv_prospective_01"
    if output.exists():
        raise SystemExit("Protocole deja present : aucun remplacement.")

    source = root / "data/csv_ssm_policy_01"
    protocol = json.loads(
        (source / "protocol.json").read_text(encoding="utf-8")
    )
    catalog = json.loads(
        (source / "catalog.json").read_text(encoding="utf-8")
    )

    artifacts = {
        "catalog": source / "catalog.json",
        "lookup": source / "lookup_policy_02.json",
        "teacher": root / "runs/csv_symbolic_teacher_01/results.json",
    }
    for seed in (97001, 97002, 97003):
        artifacts[f"ssm_{seed}"] = (
            root / f"runs/csv_ssm_policy_02/model_{seed}.pt"
        )
    for path in artifacts.values():
        if not path.is_file():
            raise SystemExit(f"Fichier absent : {path}")

    rng = random.Random(20261113)
    operations = protocol["training_behaviors"]["individual_operations"]
    compositions = rng.sample(
        list(itertools.permutations(operations, 3)), 12
    )

    probes = [test["input"] for test in catalog]
    old_cases = []
    for folder in (
        "csv_environment_01", "csv_diversity_01",
        "csv_reserved_behaviors_01", "csv_common_budget_01",
    ):
        path = root / "runs" / folder / "reserved_cases.json"
        if path.exists():
            old_cases.extend(json.loads(path.read_text(encoding="utf-8")))
    for folder in ("csv_policy_rollout_01", "csv_policy_rollout_02"):
        path = root / "runs" / folder / "evaluation_cases.json"
        if path.exists():
            old_cases.extend(json.loads(path.read_text(encoding="utf-8")))

    values = [
        "", "00041", "00092", "4.75", "-8.25",
        "e\u0301_prospective", "UPPER_prospective", "yes", "FALSE",
        "  border_new  ", "inner   gap_new", "r"*29,
        "comma,prospective", 'quote"prospective',
        "line\nprospective", "line\r\nprospective", "東京_prospective",
    ]
    cases = []
    while len(cases) < 100:
        height, width = rng.randint(1, 4), rng.randint(1, 4)
        case = [
            [rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if case not in cases and case not in probes and case not in old_cases:
            cases.append(case)

    manifest = {
        "seed": 20261113,
        "evaluation_status": "prospectively_frozen_after_previous_diagnostics",
        "independent_blind_evaluation": False,
        "scope": "new_ordered_compositions_of_three_familiar_operations",
        "compositions": compositions,
        "base_configurations": list(range(48)),
        "known_configuration_controls": 48,
        "composed_environments": 12*48,
        "maximum_calls": 8,
        "maximum_cells": 24,
        "maximum_text_bytes": 512,
        "evaluation_parameter_updates": 0,
        "checkpoint_selection": "all_three_prespecified_models",
        "equivalent_or_cancelled_compositions_must_be_reported": True,
        "no_posthoc_removal_of_difficult_or_equivalent_cases": True,
        "artifact_sha256": {
            name: {
                "path": str(path),
                "sha256": fingerprint(path),
            }
            for name, path in artifacts.items()
        },
        "cases_sha256": hashlib.sha256(
            json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }

    output.mkdir(parents=True)
    (output / "protocol.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "evaluation_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Protocole fige : 12 compositions x 48 configurations")
    print("Controles connus : 48 configurations")
    print("Tableaux reserves : 100, distincts des fichiers anterieurs disponibles")
    print("Modeles, table et catalogue : empreintes enregistrees")
    print("Aucune evaluation executee.")
    print(f"Fichiers : {output}")

if __name__ == "__main__":
    main()
