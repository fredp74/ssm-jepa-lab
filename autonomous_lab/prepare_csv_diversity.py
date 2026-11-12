import hashlib
import json
import random
from pathlib import Path

from csv_discovery import PROBES

def cell_count(probes):
    return sum(len(row) for probe in probes for row in probe)

def text_bytes(probes):
    return sum(
        len(value.encode("utf-8"))
        for probe in probes for row in probe for value in row
    )

def main():
    output = Path("autonomous_lab/data/csv_diversity_01")
    if output.exists():
        raise SystemExit("Banques deja presentes : aucun remplacement.")

    categories = {
        "numeric": ["0007", "-004", "+03", "0.50", "8e2", "007.20"],
        "whitespace": ["  trim_D  ", "inner  gap_E", "\tpad_F\t", "\u00a0pad_G\u00a0"],
        "unicode": ["e\u0301_H", "ＡＢ_I", "Straße_J", "a\u200bb_K"],
        "booleans": ["true", "FALSE", "null", "None", "NaN", "yes"],
        "punctuation": ["a|b_L", "x:y_M", "p/q_N", "r\\s_O", '"q_P"'],
        "linebreaks": ["a\nb_Q", "c\r\nd_R", "\nedge_S\n", "a\tb_T"],
        "lengths": ["x_U", "z"*32, "w"*64, ""],
    }

    # Tailles identiques aux sept premieres sondes historiques.
    sizes = [sum(len(row) for row in probe) for probe in PROBES[:7]]
    assert len(sizes) == len(categories)
    assert cell_count(PROBES) == 24

    banks = {"fixed": PROBES}
    metadata = {}
    for seed in (111001, 111002, 111003):
        rng = random.Random(seed)
        names = list(categories)
        rng.shuffle(names)
        probes = []
        assignments = []

        for size, name in zip(sizes, names):
            values = categories[name]
            cells = [rng.choice(values) for _ in range(size)]
            probes.append([cells])
            assignments.append(name)

        # Conserver la sonde qui distingue les configurations connues.
        probes.append(PROBES[7])
        assert len(probes) == 8 and cell_count(probes) == 24
        key = f"diverse_{seed}"
        banks[key] = probes
        metadata[key] = {
            "seed": seed,
            "category_by_probe": assignments + ["known_format_composite"],
        }

    for name, probes in banks.items():
        serialized = json.dumps(
            probes, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        metadata.setdefault(name, {})
        metadata[name].update({
            "calls": len(probes),
            "cells": cell_count(probes),
            "input_text_utf8_bytes": text_bytes(probes),
            "sha256": hashlib.sha256(serialized).hexdigest(),
        })
        print(f"{name} : {metadata[name]}")

    output.mkdir(parents=True)
    path = output / "probe_banks.json"
    path.write_text(
        json.dumps({
            "protocol": {
                "maximum_calls": 8,
                "cells_per_bank": 24,
                "text_length_is_reported_not_matched": True,
                "known_format_composite_preserved": True,
                "leading_zero_normalization": "development_regression_case",
                "new_unknown_behaviors_not_evaluated_yet": True,
                "neural_training": False,
            },
            "metadata": metadata,
            "banks": banks,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Banques enregistrees : {path}")

if __name__ == "__main__":
    main()
