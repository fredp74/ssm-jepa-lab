import json
import random
from pathlib import Path
import assistant_v07 as app

base = Path("runs/assistant_balanced_v07.pt")
destination = Path("instructions_v07/fresh_validation.jsonl")

if destination.exists():
    raise SystemExit("Validation deja presente : aucun remplacement.")

state = app.core.load(base)
excluded = set(state["stream_used"])
for name in ("instructions_v05/test.jsonl", "instructions_v06/test.jsonl"):
    _, values = app.answers(Path(name))
    excluded.update(values)

rng = random.Random(81008)
targets = []

# Reserver les cibles avant de generer les distracteurs.
for task, maximum in (("copie", 8), ("extraction", 5)):
    for length in range(2, maximum+1):
        for index in range(16):
            for attempt in range(10000):
                value = "".join(
                    rng.choice(app.ALPHABET) for _ in range(length)
                )
                if value not in excluded:
                    break
            else:
                raise RuntimeError(
                    f"Pas assez de cibles distinctes de longueur {length}"
                )
            excluded.add(value)
            targets.append((task, value, index % 4))

rows = []
for task, value, template in targets:
    row = app.record(value, template, task, rng, excluded)
    excluded.update(row["source_values"])
    rows.append(row)

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
    encoding="utf-8"
)
print(f"Validation neuve : {destination} | {len(rows)} exemples")
