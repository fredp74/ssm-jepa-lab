import json
from pathlib import Path

path = Path("instructions_v17/selection12.jsonl")
if path.exists():
    raise SystemExit("Fichier deja present : aucun remplacement.")

groups = [
    {"cible": "a7m", "gauche": "b2q", "droite": "c9v"},
    {"gauche": "d4x", "droite": "e8n", "cible": "f3k"},
    {"droite": "g6p", "cible": "h1z", "gauche": "j5r"},
    {"cible": "k8s", "droite": "m2w", "gauche": "n7t"},
]

rows = []
for number, fields in enumerate(groups):
    text = " ; ".join(f"{key}={value}" for key, value in fields.items())
    for requested in ("cible", "gauche", "droite"):
        rows.append({
            "task": f"selection_{requested}",
            "group_id": number,
            "source_values": list(fields.values()),
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Extrais uniquement la valeur du champ {requested} : "
                        f"{text}."
                    ),
                },
                {"role": "assistant", "content": fields[requested]},
            ],
        })

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
print(f"Diagnostic cree : {len(rows)} exemples.")
