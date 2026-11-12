import hashlib
import json
from pathlib import Path

root = Path("autonomous_lab")
original_path = root / "data/csv_diversity_01/probe_banks.json"
original = json.loads(original_path.read_text(encoding="utf-8"))

folder = root / "data/csv_contrasts_01"
trainer = root / "evaluate_csv_contrasts.py"
if folder.exists() or trainer.exists():
    raise SystemExit("Fichiers deja presents : aucun remplacement.")

contrasts = [
    [["7", "0007"]],
    [["1.25", "1,25"]],
    [["a b", "a  b "]],
    [["é_X", "e\u0301_X", "AB_Y", "ＡＢ_Y"]],
    [["yes", "TRUE"]],
    [["length_Z", "length_Z"*4]],
    [["a|b_W", "a:b_W"]],
    original["banks"]["fixed"][7],
]
assert len(contrasts) == 8
assert sum(len(row) for probe in contrasts for row in probe) == 24

banks = {
    "fixed": original["banks"]["fixed"],
    "contrasts": contrasts,
}
cases = json.loads(
    (root / "runs/csv_diversity_01/reserved_cases.json").read_text(
        encoding="utf-8"
    )
)
assert not any(
    probe == case
    for bank in banks.values()
    for probe in bank
    for case in cases
), "Une sonde reproduit un tableau reserve."

metadata = {}
for name, probes in banks.items():
    serialized = json.dumps(
        probes, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    metadata[name] = {
        "calls": len(probes),
        "cells": sum(len(row) for probe in probes for row in probe),
        "input_text_utf8_bytes": sum(
            len(value.encode("utf-8"))
            for probe in probes for row in probe for value in row
        ),
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }

source = (root / "evaluate_csv_diversity.py").read_text(
    encoding="utf-8-sig"
)
patches = [
    (
        "autonomous_lab/data/csv_diversity_01/probe_banks.json",
        "autonomous_lab/data/csv_contrasts_01/probe_banks.json",
    ),
    (
        'Path("autonomous_lab/runs/csv_diversity_01")',
        'Path("autonomous_lab/runs/csv_contrasts_01")',
    ),
    (
        "    cases = make_cases(banks)",
        '    cases = json.loads(Path(\n'
        '        "autonomous_lab/runs/csv_diversity_01/reserved_cases.json"\n'
        '    ).read_text(encoding="utf-8"))',
    ),
    (
        '        "variants_hand_designed": True,',
        '        "variants_hand_designed": True,\n'
        '        "analysis": "regression_on_previously_inspected_behaviors",\n'
        '        "evaluation_cases_reused_unchanged": True,',
    ),
]
for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

compile(source, str(trainer), "exec")
folder.mkdir(parents=True)
(folder / "probe_banks.json").write_text(
    json.dumps({
        "protocol": {
            "purpose": "deterministic_contrast_regression",
            "maximum_calls": 8,
            "cells_per_bank": 24,
            "not_an_independent_evaluation": True,
        },
        "metadata": metadata,
        "banks": banks,
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
trainer.write_text(source, encoding="utf-8")

for name, values in metadata.items():
    print(f"{name} : {values}")
print(f"Programme cree : {trainer}")
