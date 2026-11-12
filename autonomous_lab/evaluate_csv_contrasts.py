import json
import random
import re
import unicodedata
from pathlib import Path

from csv_environment import configurations, render
from csv_discovery import DiscoveryAgent

CONFIGS = configurations()
NEW_VARIANTS = (
    "unicode_NFC", "unicode_NFKC", "casefold", "rstrip",
    "collapse_spaces", "decimal_comma", "boolean_normalization",
    "truncate_16",
)

def transform(value, variant):
    if variant == "unicode_NFC":
        return unicodedata.normalize("NFC", value)
    if variant == "unicode_NFKC":
        return unicodedata.normalize("NFKC", value)
    if variant == "casefold":
        return value.casefold()
    if variant == "rstrip":
        return value.rstrip()
    if variant == "collapse_spaces":
        return re.sub(r" {2,}", " ", value)
    if variant == "decimal_comma":
        return value.replace(".", ",") if re.fullmatch(
            r"[+-]?[0-9]+\.[0-9]+", value
        ) else value
    if variant == "boolean_normalization":
        return {
            "true": "true", "false": "false",
            "yes": "true", "no": "false",
        }.get(value.lower(), value)
    if variant == "truncate_16":
        return value[:16]
    if variant == "leading_zeros_regression":
        return str(int(value)) if (
            value.isascii() and value.isdigit()
            and len(value) > 1 and value.startswith("0")
        ) else value
    return value

def hidden_render(rows, config, variant):
    return render([
        [transform(value, variant) for value in row] for row in rows
    ], config)

def make_cases(banks):
    values = [
        "", "00023", "-009", "001.25", "-7.50", "TRUE", "false",
        "yes", "no", "e\u0301_eval", "ＦＯＯ_eval", "Straße_eval",
        "start  middle", "tail   ", "  edges  ", "q"*24,
        'csv,"value"', "a;b", "a\tb", "line\nnext",
        "line\r\nnext", "🙂_eval", "a\\b",
    ]
    probes = [probe for bank in banks.values() for probe in bank]
    cases = [[[value]] for value in values]
    cases = [case for case in cases if case not in probes]
    rng = random.Random(112001)
    while len(cases) < 100:
        height, width = rng.randint(1, 3), rng.randint(1, 4)
        case = [
            [rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if case not in probes and case not in cases:
            cases.append(case)
    return cases

def main():
    source = Path("autonomous_lab/data/csv_contrasts_01/probe_banks.json")
    bank_data = json.loads(source.read_text(encoding="utf-8"))
    banks = bank_data["banks"]
    output = Path("autonomous_lab/runs/csv_contrasts_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    cases = json.loads(Path(
        "autonomous_lab/runs/csv_diversity_01/reserved_cases.json"
    ).read_text(encoding="utf-8"))
    environments = [
        ("known", index, "known") for index in range(48)
    ]
    environments += [
        ("regression", index, "leading_zeros_regression")
        for index in (0, 6, 24, 46)
    ]
    environments += [
        ("additional", index, variant)
        for index in (0, 6, 24, 46)
        for variant in NEW_VARIANTS
    ]

    output.mkdir(parents=True)
    protocol = {
        "bank_metadata": bank_data["metadata"],
        "additional_variants": list(NEW_VARIANTS),
        "regression_variant": "leading_zeros_regression",
        "outside_base_configurations": [0, 6, 24, 46],
        "reserved_cases": 100,
        "variants_hand_designed": True,
        "analysis": "regression_on_previously_inspected_behaviors",
        "evaluation_cases_reused_unchanged": True,
        "probes_frozen_before_evaluation": True,
        "stop_on_contradiction": True,
        "maximum_calls": 8,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "reserved_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = []
    for group, config_id, variant in environments:
        config = CONFIGS[config_id]
        expected = [
            hidden_render(case, config, variant) for case in cases
        ]
        if group != "known":
            assert not any(
                all(
                    render(case, candidate) == answer
                    for case, answer in zip(cases, expected)
                )
                for candidate in CONFIGS
            ), f"Comportement non distingue : {config_id}, {variant}"

        for bank_name, probes in banks.items():
            agent = DiscoveryAgent()
            calls = cells = text_bytes = 0
            for probe in probes:
                response = hidden_render(probe, config, variant)
                calls += 1
                cells += sum(len(row) for row in probe)
                text_bytes += sum(
                    len(value.encode("utf-8"))
                    for row in probe for value in row
                )
                agent.observe(probe, response)
                if not agent.candidates:
                    break

            predictions = [agent.predict(case) for case in cases]
            exact = sum(
                prediction is not None and prediction == answer
                for prediction, answer in zip(predictions, expected)
            )
            results.append({
                "group": group,
                "variant": variant,
                "configuration": config_id,
                "bank": bank_name,
                "detected_unknown": not agent.candidates,
                "remaining_candidates": len(agent.candidates),
                "calls": calls,
                "cells": cells,
                "input_text_utf8_bytes": text_bytes,
                "answered": sum(p is not None for p in predictions),
                "exact": exact,
                "test_cases": len(cases),
            })

    summaries = []
    for bank_name in banks:
        print(f"\nBANQUE : {bank_name}")
        for group in ("known", "regression", "additional"):
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

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": protocol,
            "summaries": summaries,
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
