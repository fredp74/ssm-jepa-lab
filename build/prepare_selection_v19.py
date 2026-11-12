import json
import random
from pathlib import Path

root = Path("instructions_v19")
outputs = [root / "selection.jsonl", root / "mixed.jsonl"]
if any(path.exists() for path in outputs):
    raise SystemExit("Donnees v19 deja presentes : aucun remplacement.")

def read(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

selection = read("instructions_v18/train.jsonl")
audit = read("runs/experiment_v13_control_130015.train_audit.jsonl")
rng = random.Random(190019)
old = []

for prefix, count in (
    ("copie_", 32),
    ("extraction_", 16),
    ("dialogue_", 16),
    ("mise_a_jour_", 16),
    ("inconnu_", 16),
):
    pool = [row for row in audit if row["task"].startswith(prefix)]
    if len(pool) < count:
        raise SystemExit(f"Audit insuffisant pour {prefix} : {len(pool)}")
    old.extend(rng.sample(pool, count))

if len(selection) != 96 or len(old) != 96:
    raise SystemExit("Nombre d'exemples inattendu.")

mixed = selection + old
rng.shuffle(mixed)
root.mkdir(parents=True, exist_ok=True)

for name, rows in (("selection", selection), ("mixed", mixed)):
    path = root / f"{name}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"{path} : {len(rows)} exemples")
