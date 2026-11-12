import itertools
import json
import random
import string
from pathlib import Path

root = Path("instructions_v22")
if any((root / name).exists() for name in ("train.jsonl", "validation.jsonl")):
    raise SystemExit("Donnees deja presentes : aucun remplacement.")

rng = random.Random(220022)
alphabet = string.ascii_lowercase + string.digits
names = ("cible", "gauche", "droite")
used_sets = set()
datasets = {}

for split, count in (("train", 32), ("validation", 16)):
    rows = []
    for group in range(count):
        while True:
            characters = rng.sample(alphabet, 3)
            identity = tuple(sorted(characters))
            if identity not in used_sets:
                used_sets.add(identity)
                break

        values = dict(zip(names, characters))

        for order in itertools.permutations(names):
            text = " ; ".join(f"{name}={values[name]}" for name in order)
            for requested in names:
                rows.append({
                    "task": f"selection_{requested}",
                    "group_id": group,
                    "order": list(order),
                    "source_values": characters,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Extrais uniquement la valeur du champ {requested} : "
                                f"{text}."
                            ),
                        },
                        {"role": "assistant", "content": values[requested]},
                    ],
                })

    rng.shuffle(rows)
    datasets[split] = rows

root.mkdir(parents=True, exist_ok=True)
for split, rows in datasets.items():
    path = root / f"{split}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"{path} : {len(rows)} exemples")
