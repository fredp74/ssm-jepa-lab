import csv
import hashlib
import io
import json
import random
from pathlib import Path

from csv_environment import configurations, render
from csv_discovery import DiscoveryAgent

CONFIGS = configurations()
VARIANTS = (
    "deduplicate_rows", "drop_first_row", "rotate_columns",
    "sort_columns", "drop_empty_columns", "escape_formulas",
    "normalize_CRLF", "append_row_number",
)

def hidden_render(rows, config, variant):
    values = [list(row) for row in rows]
    if variant == "deduplicate_rows":
        seen = set()
        values = [
            row for row in values
            if not (tuple(row) in seen or seen.add(tuple(row)))
        ]
    elif variant == "drop_first_row":
        values = values[1:]
    elif variant == "rotate_columns":
        values = [row[1:] + row[:1] for row in values]
    elif variant == "sort_columns":
        order = sorted(range(len(values[0])), key=lambda i: values[0][i])
        values = [[row[i] for i in order] for row in values]
    elif variant == "drop_empty_columns":
        keep = [
            i for i in range(len(values[0]))
            if any(row[i] != "" for row in values)
        ]
        values = [[row[i] for i in keep] for row in values] if keep else []
    elif variant == "escape_formulas":
        values = [
            [
                "'" + value if value.startswith(("=", "+", "-", "@"))
                else value
                for value in row
            ]
            for row in values
        ]
    elif variant == "normalize_CRLF":
        values = [
            [value.replace("\r\n", "\n") for value in row]
            for row in values
        ]
    elif variant == "append_row_number":
        values = [row + [str(i+1)] for i, row in enumerate(values)]
    return render(values, config) if values else ""

def relation(rows, operation):
    if operation == "reverse_rows":
        return list(reversed(rows))
    if operation == "duplicate_rows":
        return rows + rows
    if operation == "append_empty_row":
        return rows + [[""]*len(rows[0])]
    if operation == "prepend_empty_row":
        return [[""]*len(rows[0])] + rows
    if operation == "transpose":
        return [list(column) for column in zip(*rows)]
    raise ValueError(operation)

def parse(text, config):
    return list(csv.reader(
        io.StringIO(text, newline=""),
        delimiter=config["delimiter"],
        quotechar='"',
        doublequote=config["doublequote"],
        escapechar=None if config["doublequote"] else "\\",
        strict=True,
    ))

def make_cases(plans):
    probes = [call["input"] for plan in plans.values() for call in plan]
    cases = [
        [["dup_new", "v"], ["dup_new", "v"]],
        [["z_new", "a_new"], ["2", "1"]],
        [["", "kept_new"], ["", "other_new"]],
        [["=SUM(A1:A2)", "+123"], ["-45", "@name"]],
        [["first_new\r\nsecond_new"]],
    ]
    assert not any(case in probes for case in cases)

    rng = random.Random(114001)
    pool = [
        "", "new_alpha", "new_beta", "009_new", "=A1+1",
        "+12", "-7", "@user", " padded_new ", '"quoted_new"',
        "semi;new", "tab\tnew", "x\r\ny_new", "x\ny_new", "été_new",
    ]
    while len(cases) < 100:
        height, width = rng.randint(1, 4), rng.randint(1, 4)
        case = [
            [rng.choice(pool) for _ in range(width)]
            for _ in range(height)
        ]
        if case not in cases and case not in probes:
            cases.append(case)
    return cases

def main():
    source = Path("autonomous_lab/data/csv_common_budget_01/plans.json")
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    plans = data["plans"]
    for name, plan in plans.items():
        encoded = json.dumps(
            plan, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        assert hashlib.sha256(encoded).hexdigest() == data["metadata"][name]["sha256"]

    output = Path("autonomous_lab/runs/csv_common_budget_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    cases = make_cases(plans)
    environments = [
        ("known", i, "known") for i in range(48)
    ] + [
        ("additional", i, variant)
        for i in (2, 14, 28, 42) for variant in VARIANTS
    ]
    output.mkdir(parents=True)
    config = {
        "plans_sha256": hashlib.sha256(raw).hexdigest(),
        "plan_metadata": data["metadata"],
        "variants": list(VARIANTS),
        "maximum_calls": 8,
        "maximum_cells": 24,
        "maximum_text_bytes": 512,
        "evaluation_cases": 100,
        "variants_hand_designed": True,
        "independent_blind_evaluation": False,
        "seed": 114001,
    }
    (output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "reserved_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results = []
    for group, config_id, variant in environments:
        expected = [
            hidden_render(case, CONFIGS[config_id], variant) for case in cases
        ]
        if group != "known":
            assert not any(
                all(render(case, c) == answer for case, answer in zip(cases, expected))
                for c in CONFIGS
            ), f"Variante non distinguee : {variant}"

        for name, plan in plans.items():
            agent = DiscoveryAgent()
            outputs = {}
            calls = cells = text_bytes = 0
            detected = False

            for call_id, call in enumerate(plan):
                rows = call["input"]
                calls += 1
                cells += sum(len(row) for row in rows)
                text_bytes += sum(
                    len(v.encode("utf-8")) for row in rows for v in row
                )
                assert calls <= 8 and cells <= 24 and text_bytes <= 512
                answer = hidden_render(rows, CONFIGS[config_id], variant)
                outputs[call_id] = answer

                if call["check"] == "decoded_relation":
                    if len(agent.candidates) != 1:
                        raise RuntimeError("Format ambigu pour le controle relationnel.")
                    inferred = CONFIGS[agent.candidates[0]]
                    try:
                        baseline = parse(outputs[call["baseline_call"]], inferred)
                        detected = parse(answer, inferred) != relation(
                            baseline, call["operation"]
                        )
                    except csv.Error:
                        detected = True
                    if detected:
                        break

                agent.observe(rows, answer)
                if not agent.candidates:
                    detected = True
                    break

            predictions = (
                [None]*len(cases) if detected
                else [agent.predict(case) for case in cases]
            )
            results.append({
                "group": group, "variant": variant,
                "configuration": config_id, "plan": name,
                "detected_unknown": detected,
                "calls": calls, "cells": cells, "text_bytes": text_bytes,
                "answered": sum(p is not None for p in predictions),
                "exact": sum(
                    p is not None and p == y
                    for p, y in zip(predictions, expected)
                ),
            })

    summaries = []
    for name in plans:
        print(f"\nPLAN : {name}")
        for group in ("known", "additional"):
            rows = [r for r in results if r["plan"] == name and r["group"] == group]
            summary = {
                "plan": name, "group": group, "environments": len(rows),
                "detected_unknown": sum(r["detected_unknown"] for r in rows),
                "fully_correct": sum(r["exact"] == 100 for r in rows),
                "incorrect_survivors": sum(
                    not r["detected_unknown"] and r["exact"] < 100 for r in rows
                ),
                "mean_calls": sum(r["calls"] for r in rows)/len(rows),
                "mean_cells": sum(r["cells"] for r in rows)/len(rows),
            }
            summaries.append(summary)
            print(summary)

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": config, "summaries": summaries, "results": results
        }, indent=2), encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
