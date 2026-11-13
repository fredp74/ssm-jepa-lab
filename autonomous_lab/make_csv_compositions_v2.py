import ast
from pathlib import Path

source = Path("autonomous_lab/acquire_csv_compositions.py")
destination = Path("autonomous_lab/acquire_csv_compositions_v2.py")
if destination.exists():
    raise SystemExit("Destination existante : aucun remplacement.")

text = source.read_text(encoding="utf-8-sig")


def replace_once(old, new):
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"Structure inattendue : {old!r} trouve {count} fois. "
            "Aucun script cree."
        )
    text = text.replace(old, new, 1)


replace_once(
    'output = root / "runs/csv_composition_01"',
    'output = root / "runs/csv_composition_02"',
)

replace_once(
    '''    checks = [
        [["A", "00042"], ["a", "00042"]],
        [["  ", ""], ["e\\u0301_verify ", "7.25"]],
    ]''',
    '''    learning_checks = [
        [["A", "00042"], ["a", "00042"]],
        [["  ", ""], ["e\\u0301_verify ", "7.25"]],
    ]
    checks = [
        [["abcdefghijklmnop_X", "Q"],
         ["abcdefghijklmnop_Y", "Q"]],
        [["e\\u0301_check ", "0048"], ["é_check ", "0048"]],
    ]''',
)

replace_once(
    '"verification_inputs": checks,',
    '''"learning_inputs": learning_checks,
        "verification_inputs": checks,
        "allocation": {
            "catalogue": {"calls": 4, "cells": 8, "bytes": 256},
            "learning": {"calls": 2, "cells": 8, "bytes": 128},
            "verification": {"calls": 2, "cells": 8, "bytes": 128},
        },
        "development_environments_reused": True,''',
)

replace_once(
    '    records = []',
    '''    records = []
    saved_snapshots = {}''',
)

replace_once(
    'while survivors and len(used) < 6:',
    'while survivors and len(used) < 4:',
)
replace_once(
    'cells + test["cells"] <= 16',
    'cells + test["cells"] <= 8',
)
replace_once(
    'text_bytes + test["input_text_utf8_bytes"] <= 384',
    'text_bytes + test["input_text_utf8_bytes"] <= 256',
)

replace_once(
    '            # Figer les hypotheses avant les verifications.',
    '''            # Deux observations d'apprentissage avant de figer.
            learning_observations = []
            for rows in learning_checks:
                if not survivors:
                    break
                answer = query(rows)
                survivors = [
                    i for i in survivors
                    if execute(programs[i], rows) == answer
                ]
                learning_observations.append({
                    "input": rows, "output": answer,
                    "remaining_programs": len(survivors),
                })

            # Figer les hypotheses avant les verifications.''',
)

replace_once(
    '            write(memory_path, memory)',
    '''            persistence_inputs = [
                test["input"] for test in catalog
            ] + checks
            saved_snapshots[str(memory_path)] = (
                json.dumps(memory, sort_keys=True, ensure_ascii=False),
                [
                    execute(program, rows)
                    for program in memory["programs"]
                    for rows in persistence_inputs
                ],
            )
            write(memory_path, memory)''',
)

replace_once(
    '"selected_test_ids": used,',
    '''"selected_test_ids": used,
                "learning_observations": learning_observations,''',
)

replace_once('random.Random(119001)', 'random.Random(120001)')
replace_once('"evaluation_seed": 119001,', '"evaluation_seed": 120001,')
replace_once(
    '"same_prefix_123456_X", "same_prefix_123456_Y",',
    '"shared_prefix_abcdefghijkl_X", "shared_prefix_abcdefghijkl_Y",',
)
replace_once(
    'forbidden = [t["input"] for t in catalog] + checks',
    '''forbidden = (
        [t["input"] for t in catalog] + checks + learning_checks
        + read(root / "runs/csv_composition_01/evaluation_cases.json")
    )''',
)

replace_once(
    'memory_roundtrip_passed=(restored == read(memory_path)),',
    '''memory_roundtrip_passed=(
                json.dumps(restored, sort_keys=True, ensure_ascii=False)
                == saved_snapshots[str(memory_path)][0]
                and [
                    execute(program, rows)
                    for program in restored["programs"]
                    for rows in (
                        [test["input"] for test in catalog] + checks
                    )
                ] == saved_snapshots[str(memory_path)][1]
            ),''',
)

replace_once(
    '    summaries = []',
    '''    if not all(r["memory_roundtrip_passed"] for r in records):
        raise RuntimeError("Echec du controle de persistance.")
    summaries = []''',
)

ast.parse(text)
destination.write_text(text, encoding="utf-8")
print(f"Cree : {destination}")
print("Script valide syntaxiquement ; aucune experience executee.")
