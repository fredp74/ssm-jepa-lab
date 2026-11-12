import json
import random
import string
from pathlib import Path

root = Path("instructions_v03")
outputs = [
    root/f"{stage}_{split}.jsonl"
    for stage in ("simple", "dialogue")
    for split in ("train", "test")
]
if any(path.exists() for path in outputs):
    raise SystemExit("Donnees v03 deja presentes : aucun remplacement.")
root.mkdir(exist_ok=True)
rng = random.Random(43003)
used = set()

def word(length):
    while True:
        value = "".join(
            rng.choice(string.ascii_lowercase+string.digits)
            for _ in range(length)
        )
        if value not in used:
            used.add(value)
            return value

def example(task, question, answer):
    return {"task": task, "messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}

datasets = {}
for split, count in (("train", 256), ("test", 64)):
    rows = []
    for i in range(count):
        length = 2+i % 7
        payload = word(length)
        question = rng.choice([
            f"Recopie uniquement le texte entre crochets : [{payload}].",
            f"Reponds seulement avec le contenu de [{payload}].",
            f"Copie sans commentaire : [{payload}].",
            f"Quel texte est entre crochets ? [{payload}] Reponds seulement avec ce texte.",
        ])
        rows.append(example("copie", question, payload))

        payload = word(length)
        fields = [
            f"gauche={word(length)}",
            f"cible={payload}",
            f"droite={word(length)}",
        ]
        rng.shuffle(fields)
        question = (
            "Extrais uniquement la valeur du champ cible : "
            + " ; ".join(fields) + "."
        )
        rows.append(example("extraction", question, payload))
    rng.shuffle(rows)
    datasets[f"simple_{split}"] = rows

for split, count in (("train", 128), ("test", 32)):
    rows = []
    for i in range(count):
        payload = word(2+i % 7)
        rows.append({"task": "rappel_dialogue", "messages": [
            {
                "role": "user",
                "content": f"Retiens ce code : {payload}. Reponds seulement OK."
            },
            {"role": "assistant", "content": "OK"},
            {
                "role": "user",
                "content": "Quel code viens-je de donner ? Reponds seulement avec ce code."
            },
            {"role": "assistant", "content": payload},
        ]})
    datasets[f"dialogue_{split}"] = rows

for name, records in datasets.items():
    path = root/f"{name}.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records),
        encoding="utf-8"
    )
    print(f"{path} : {len(records)} exemples")

print("Chaines distinctes ; memes familles et longueurs train/test.")
