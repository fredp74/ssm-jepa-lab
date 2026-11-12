from pathlib import Path

source = Path("autonomous_lab/context_holdout.py").read_text(
    encoding="utf-8-sig"
)
destination = Path("autonomous_lab/context_holdout_ssm.py")
if destination.exists():
    raise SystemExit("Programme deja present : aucun remplacement.")

patches = [
    (
        "from keyed_memory import KeyedMemory",
        "from meta_ssm import MetaSSM",
    ),
    (
        'Path("autonomous_lab/runs/context_holdout_01")',
        'Path("autonomous_lab/runs/context_holdout_ssm_01")',
    ),
    (
        '"architecture": "exact_keyed_memory_plus_mlp"',
        '"architecture": "diagonal_ssm"',
    ),
    (
        "        model = KeyedMemory()",
        "        model = MetaSSM()",
    ),
]

for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print(f"Programme cree : {destination}")
