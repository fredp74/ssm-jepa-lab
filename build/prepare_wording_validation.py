import copy
import json
import re
from pathlib import Path

destination = Path("instructions_v14/wording_validation.jsonl")
if destination.exists():
    raise SystemExit("Validation deja presente : aucun remplacement.")

source = Path("instructions_v11/validation.jsonl")
records = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]

rows = []
for number, row in enumerate(records):
    task = row["task"]
    question = row["messages"][0]["content"]

    if task.startswith("copie_"):
        match = re.search(r"\[([a-z0-9]+)\]", question)
        if not match:
            raise ValueError(f"Crochets introuvables : {question}")
        value = match.group(1)
        rewritten = (
            f"Voici une chaine : [{value}]. "
            "Ecris exactement cette chaine, sans les crochets ni autre texte."
        )
    elif task.startswith("extraction_"):
        fields = re.findall(
            r"(cible|gauche|droite)=([a-z0-9]+)", question
        )
        if len(fields) != 3:
            raise ValueError(f"Champs inattendus : {question}")
        text = " ; ".join(f"{key}={value}" for key, value in fields)
        rewritten = (
            f"Champs disponibles : {text}. "
            "Ecris seulement ce qui est affecte a cible."
        )
    else:
        continue

    for group, prompt in (
        ("original", question),
        ("reformule", rewritten),
    ):
        item = copy.deepcopy(row)
        item["task"] = f"{group}_{task}"
        item["pair_id"] = number
        item["messages"][0]["content"] = prompt
        rows.append(item)

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
print(f"Validation creee : {len(rows)} cas, soit {len(rows)//2} paires.")
