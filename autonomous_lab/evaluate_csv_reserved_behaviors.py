import hashlib
import json
import random
from pathlib import Path

from csv_environment import configurations, render
from csv_discovery import DiscoveryAgent

CONFIGS = configurations()
VARIANTS = (
    "reverse_rows",
    "reverse_columns",
    "transpose",
    "sort_rows",
    "generated_header",
    "remove_empty_rows",
    "omit_final_line_ending",
    "expand_tabs",
)

def hidden_render(rows, config, variant):
    values = [list(row) for row in rows]

    if variant == "reverse_rows":
        values.reverse()
    elif variant == "reverse_columns":
        values = [list(reversed(row)) for row in values]
    elif variant == "transpose":
        values = [list(column) for column in zip(*values)]
    elif variant == "sort_rows":
        values.sort()
    elif variant == "generated_header":
        values = [
            [f"column_{i+1}" for i in range(len(values[0]))]
        ] + values
    elif variant == "remove_empty_rows":
        values = [row for row in values if any(value != "" for value in row)]
    elif variant == "expand_tabs":
        values = [
            [value.replace("\t", "    ") for value in row]
            for row in values
        ]

    result = render(values, config) if values else ""
    if variant == "omit_final_line_ending" and result:
        result = result[:-len(config["line_ending"])]
    return result

def make_cases(banks):
    probes = [probe for bank in banks.values() for probe in bank]
    old_cases = json.loads(Path(
        "autonomous_lab/runs/csv_diversity_01/reserved_cases.json"
    ).read_text(encoding="utf-8"))

    candidates = [
        [["", ""]],
        [["row_B", "two"], ["row_A", "one"]],
        [["left_C", "right_D", "last_E"]],
        [["tab\tinside_F", "plain_G"]],
        [["", ""], ["data_H", "value_I"]],
    ]
    cases = [
        case for case in candidates
        if case not in probes and case not in old_cases
    ]

    rng = random.Random(113001)
    values = [
        "", "alpha_new", "beta_new", "007_new", "12.5_new",
        " padded_new ", 'quote"new', "comma,new", "semi;new",
        "tab\tnew", "line\nnew", "line\r\nnew", "été_new", "東京_new",
    ]
    while len(cases) < 100:
        height, width = rng.randint(1, 4), rng.randint(1, 4)
        case = [
            [rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if case not in probes and case not in old_cases and case not in cases:
            cases.append(case)
    return cases

def main():
    bank_path = Path("autonomous_lab/data/csv_contrasts_01/probe_banks.json")
    raw = bank_path.read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    banks = frozen["banks"]

    for name, probes in banks.items():
        serialized = json.dumps(
            probes, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        assert hashlib.sha256(serialized).hexdigest() == (
            frozen["metadata"][name]["sha256"]
        )

    output = Path("autonomous_lab/runs/csv_reserved_behaviors_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    cases = make_cases(banks)
    environments = [
        ("known", index, "known") for index in range(48)
    ] + [
        ("additional", index, variant)
        for index in (2, 14, 28, 42)
        for variant in VARIANTS
    ]
    config = {
        "frozen_bank_file_sha256": hashlib.sha256(raw).hexdigest(),
        "bank_metadata": frozen["metadata"],
        "additional_variants": list(VARIANTS),
        "outside_base_configurations": [2, 14, 28, 42],
        "new_evaluation_cases": 100,
        "maximum_calls": 8,
        "probe_changes": False,
        "variants_hand_designed": True,
        "independent_blind_evaluation": False,
        "seed": 113001,
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "reserved_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = []
    for group, config_id, variant in environments:
        options = CONFIGS[config_id]
        expected = [
            hidden_render(case, options, variant) for case in cases
        ]
        if group != "known":
            assert not any(
                all(
                    render(case, candidate) == answer
                    for case, answer in zip(cases, expected)
                )
                for candidate in CONFIGS
            ), f"Variante non distinguee : {variant}, {config_id}"

        for bank_name, probes in banks.items():
            agent = DiscoveryAgent()
            calls = 0
            for probe in probes:
                response = hidden_render(probe, options, variant)
                calls += 1
                agent.observe(probe, response)
                if not agent.candidates:
                    break

            predictions = [agent.predict(case) for case in cases]
            results.append({
                "group": group,
                "configuration": config_id,
                "variant": variant,
                "bank": bank_name,
                "calls": calls,
                "detected_unknown": not agent.candidates,
                "remaining_candidates": len(agent.candidates),
                "answered": sum(value is not None for value in predictions),
                "exact": sum(
                    prediction is not None and prediction == answer
                    for prediction, answer in zip(predictions, expected)
                ),
                "cases": len(cases),
            })

    summaries = []
    for bank_name in banks:
        print(f"\nBANQUE : {bank_name}")
        for group in ("known", "additional"):
            rows = [
                row for row in results
                if row["bank"] == bank_name and row["group"] == group
            ]
            summary = {
                "bank": bank_name,
                "group": group,
                "environments": len(rows),
                "detected_unknown": sum(r["detected_unknown"] for r in rows),
                "fully_correct": sum(r["exact"] == 100 for r in rows),
                "incorrect_survivors": sum(
                    not r["detected_unknown"] and r["exact"] < 100
                    for r in rows
                ),
                "mean_calls": sum(r["calls"] for r in rows)/len(rows),
            }
            summaries.append(summary)
            print(summary)

        for variant in VARIANTS:
            rows = [
                row for row in results
                if row["bank"] == bank_name and row["variant"] == variant
            ]
            print(
                f"{variant} : "
                f"{sum(row['detected_unknown'] for row in rows)}/{len(rows)}"
            )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": config,
            "summaries": summaries,
            "results": results,
        }, indent=2), encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
