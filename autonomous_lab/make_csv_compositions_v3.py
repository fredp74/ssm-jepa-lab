import ast
from pathlib import Path

source = Path("autonomous_lab/acquire_csv_compositions_v2.py")
destination = Path("autonomous_lab/acquire_csv_compositions_v3.py")
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
    'output = root / "runs/csv_composition_02"',
    'output = root / "runs/csv_composition_03"',
)

replace_once(
    '''    checks = [
        [["abcdefghijklmnop_X", "Q"],
         ["abcdefghijklmnop_Y", "Q"]],
        [["e\\u0301_check ", "0048"], ["é_check ", "0048"]],
    ]''',
    '''    checks = [
        [
            ["z_verify"],
            ["a_verify"],
            ["abcdefghijklmnop_X"],
            ["abcdefghijklmnop_Y"],
        ],
        [
            ["e\\u0301_check "],
            ["é_check "],
            ["   "],
            ["00048"],
        ],
    ]''',
)

replace_once(
    '"development_environments_reused": True,',
    '''"development_environments_reused": True,
        "verification_inputs_redesigned_after_v2": True,
        "verification_targets": [
            "row_order",
            "collision_after_truncation",
            "unicode_normalization",
            "whitespace_and_empty_rows",
            "leading_zeros",
        ],''',
)

replace_once('random.Random(120001)', 'random.Random(121001)')
replace_once('"evaluation_seed": 120001,', '"evaluation_seed": 121001,')

replace_once(
    '+ read(root / "runs/csv_composition_01/evaluation_cases.json")',
    '''+ read(root / "runs/csv_composition_01/evaluation_cases.json")
        + read(root / "runs/csv_composition_02/evaluation_cases.json")''',
)

ast.parse(text)
destination.write_text(text, encoding="utf-8")
print(f"Cree : {destination}")
print("Script valide syntaxiquement ; aucune experience executee.")
