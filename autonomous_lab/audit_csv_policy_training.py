import hashlib
import json
from collections import Counter
from pathlib import Path

from csv_environment import configurations, render
from csv_policy_behaviors import simulate

def main():
    source = Path("autonomous_lab/data/csv_ssm_policy_01")
    protocol = json.loads(
        (source / "protocol.json").read_text(encoding="utf-8")
    )
    catalog = json.loads(
        (source / "catalog.json").read_text(encoding="utf-8")
    )
    output = Path("autonomous_lab/runs/csv_policy_training_audit_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    configs = configurations()
    operations = ["identity"] + protocol[
        "training_behaviors"
    ]["individual_operations"]

    known_outputs = [
        [render(test["input"], config) for config in configs]
        for test in catalog
    ]
    full_mask = (1 << len(configs))-1

    def feasible(indices):
        return (
            len(indices) <= protocol["maximum_calls"]
            and sum(catalog[i]["cells"] for i in indices)
                <= protocol["maximum_cells"]
            and sum(catalog[i]["input_text_utf8_bytes"] for i in indices)
                <= protocol["maximum_input_text_utf8_bytes"]
        )

    singles = sorted(
        (i for i in range(len(catalog)) if feasible([i])),
        key=lambda i: (
            catalog[i]["cells"], catalog[i]["input_text_utf8_bytes"]
        ),
    )
    pairs = [
        (i, j) for i in range(len(catalog))
        for j in range(i+1, len(catalog))
        if feasible([i, j])
    ]

    results = []
    signatures = Counter()

    for operation in operations:
        for config_id, config in enumerate(configs):
            responses = [
                simulate(test["input"], config, [operation])
                for test in catalog
            ]
            masks = [
                sum(
                    1 << candidate
                    for candidate, expected in enumerate(known_outputs[i])
                    if expected != response
                )
                for i, response in enumerate(responses)
            ]

            witness = next(
                ([i] for i in singles if masks[i] == full_mask), None
            )
            if witness is None:
                witness = next(
                    ([i, j] for i, j in pairs
                     if masks[i] | masks[j] == full_mask),
                    None,
                )

            union = 0
            for mask in masks:
                union |= mask

            if witness is not None:
                status = "budget_feasible_witness"
            elif union != full_mask:
                status = "matches_known_on_entire_catalog"
            else:
                status = "budget_feasibility_unresolved"

            digest = hashlib.sha256(
                json.dumps(responses, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            signatures[digest] += 1
            results.append({
                "operation": operation,
                "configuration": config_id,
                "status": status,
                "witness_test_ids": witness,
                "catalog_response_sha256": digest,
            })

    summaries = []
    for operation in operations:
        rows = [row for row in results if row["operation"] == operation]
        summary = {
            "operation": operation,
            "environments": len(rows),
            "statuses": dict(Counter(row["status"] for row in rows)),
        }
        summaries.append(summary)
        print(summary)

    summary = {
        "training_environments": len(results),
        "distinct_catalog_response_patterns": len(signatures),
        "reserved_compositions_evaluated": False,
        "witness_search": "one_or_two_tests_only",
        "larger_budget_search_performed": False,
    }
    print("\nBILAN :", summary)

    output.mkdir(parents=True)
    path = output / "results.json"
    path.write_text(
        json.dumps({
            "summary": summary,
            "summaries": summaries,
            "results": results,
        }, indent=2), encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
