import csv
import io
import json
from pathlib import Path

from csv_environment import configurations
from csv_discovery import DiscoveryAgent, PROBES
from evaluate_csv_reserved_behaviors import hidden_render, VARIANTS

CONFIGS = configurations()

def parse(text, config):
    return list(csv.reader(
        io.StringIO(text, newline=""),
        delimiter=config["delimiter"],
        quotechar='"',
        doublequote=config["doublequote"],
        escapechar=None if config["doublequote"] else "\\",
        strict=True,
    ))

def operations():
    return [
        ("reverse_rows", lambda rows: list(reversed(rows))),
        ("reverse_columns", lambda rows: [list(reversed(row)) for row in rows]),
        ("duplicate_rows", lambda rows: rows + rows),
        ("append_empty_row", lambda rows: rows + [[""]*len(rows[0])]),
        ("prepend_empty_row", lambda rows: [[""]*len(rows[0])] + rows),
        ("transpose", lambda rows: [list(column) for column in zip(*rows)]),
    ]

def main():
    output = Path("autonomous_lab/runs/csv_relational_checks_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    environments = [
        ("known", index, "known") for index in range(48)
    ] + [
        ("additional", index, variant)
        for index in (2, 14, 28, 42)
        for variant in VARIANTS
    ]

    results = []
    for group, config_id, variant in environments:
        actual_config = CONFIGS[config_id]
        calls = []

        def query(rows):
            if len(calls) >= 8:
                raise RuntimeError("Budget epuise.")
            answer = hidden_render(rows, actual_config, variant)
            calls.append({
                "cells": sum(len(row) for row in rows),
                "input_bytes": sum(
                    len(value.encode("utf-8"))
                    for row in rows for value in row
                ),
            })
            return answer

        agent = DiscoveryAgent()
        probe = PROBES[7]
        agent.observe(probe, query(probe))
        detected = not agent.candidates
        stage = "configuration_probe" if detected else None
        violations = []

        if agent.candidates:
            if len(agent.candidates) != 1:
                raise RuntimeError("Format ambigu : relations non interpretees.")

            inferred = CONFIGS[agent.candidates[0]]
            base = [["row_B,field", "value_2"], ["row_A", "value_1"]]
            try:
                decoded_base = parse(query(base), inferred)
            except csv.Error:
                detected = True
                stage = "base_parse"
            else:
                for name, operation in operations():
                    changed_input = operation(base)
                    expected_relation = operation(decoded_base)
                    try:
                        actual_relation = parse(query(changed_input), inferred)
                        violation = actual_relation != expected_relation
                    except csv.Error:
                        actual_relation = None
                        violation = True

                    if violation:
                        detected = True
                        stage = "relational_check"
                        violations.append({
                            "relation": name,
                            "expected": expected_relation,
                            "obtained": actual_relation,
                        })

        results.append({
            "group": group,
            "variant": variant,
            "configuration": config_id,
            "detected_violation": detected,
            "detection_stage": stage,
            "violations": violations,
            "calls": len(calls),
            "cells": sum(row["cells"] for row in calls),
            "input_bytes": sum(row["input_bytes"] for row in calls),
        })

    summaries = []
    for group in ("known", "additional"):
        rows = [row for row in results if row["group"] == group]
        summary = {
            "group": group,
            "environments": len(rows),
            "detected_violation": sum(row["detected_violation"] for row in rows),
            "mean_calls": sum(row["calls"] for row in rows)/len(rows),
            "mean_cells": sum(row["cells"] for row in rows)/len(rows),
        }
        summaries.append(summary)
        print(summary)

    for variant in VARIANTS:
        rows = [row for row in results if row["variant"] == variant]
        print(
            f"{variant} : "
            f"{sum(row['detected_violation'] for row in rows)}/{len(rows)}"
        )

    output.mkdir(parents=True)
    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "maximum_calls": 8,
                "relations": [name for name, _ in operations()],
                "comparison": "decoded_output_relation",
                "same_cell_budget_as_previous_banks": False,
                "previously_inspected_behaviors": True,
                "neural_training": False,
            },
            "summaries": summaries,
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
