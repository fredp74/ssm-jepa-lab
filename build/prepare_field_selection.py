import json
import re
from pathlib import Path

destination = Path("instructions_v15/field_selection.jsonl")
if destination.exists():
    raise SystemExit("Test deja present : aucun remplacement.")

source = Path("instructions_v11/validation.jsonl")
rows = []

for number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines()):
    if not line.strip():
        continue
    record = json.loads(line)
    if not record["task"].startswith("extraction_"):
        continue

    question = record["messages"][0]["content"]
    fields = re.findall(r"(cible|gauche|droite)=([a-z0-9]+)", question)
    values = dict(fields)
    if set(values) != {"cible", "gauche", "droite"}:
        raise ValueError(f"Champs inattendus : {question}")
    if len(set(values.values())) != 3:
        raise ValueError("Les trois valeurs doivent etre distinctes.")

    text = " ; ".join(f"{key}={value}" for key, value in fields)

    for requested in ("cible", "gauche", "droite"):
        rows.append({
            "task": f"demande_{requested}",
            "pair_id": number,
            "requested_field": requested,
            "field_values": values,
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

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
print(f"Test cree : {len(rows)} cas, soit {len(rows)//3} triplets.")
