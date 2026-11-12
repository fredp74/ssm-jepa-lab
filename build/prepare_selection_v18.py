import json
import random
import string
from pathlib import Path
import assistant_v09 as trainer

root = Path("instructions_v18")
outputs = [root / "train.jsonl", root / "validation.jsonl"]
if any(path.exists() for path in outputs):
    raise SystemExit("Donnees v18 deja presentes : aucun remplacement.")

alphabet = string.ascii_lowercase + string.digits
reserved = set()

for checkpoint in (
    "runs/experiment_v13_control_130015.pt",
    "runs/experiment_v16_selection.pt",
):
    state = trainer.core.load(Path(checkpoint))
    reserved.update(state.get("stream_used", []))

paths = list(Path(".").glob("instructions_*/*.jsonl"))
paths += list(Path("final_evaluation").glob("*.jsonl"))
for path in paths:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            reserved.update(row.get("source_values", []))

rng = random.Random(180018)

def new_code():
    for _ in range(100000):
        value = "".join(rng.choice(alphabet) for _ in range(3))
        if value not in reserved:
            reserved.add(value)
            return value
    raise RuntimeError("Generation epuisee.")

datasets = {}
for split, count in (("train", 32), ("validation", 16)):
    rows = []
    for group in range(count):
        fields = [(name, new_code()) for name in ("cible", "gauche", "droite")]
        rng.shuffle(fields)
        values = dict(fields)
        text = " ; ".join(f"{name}={value}" for name, value in fields)

        for requested in ("cible", "gauche", "droite"):
            rows.append({
                "task": f"selection_{requested}",
                "group_id": group,
                "source_values": list(values.values()),
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
    datasets[split] = rows

root.mkdir(parents=True, exist_ok=True)
for split, rows in datasets.items():
    path = root / f"{split}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"{path} : {len(rows)} exemples")
