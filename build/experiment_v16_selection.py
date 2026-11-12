import sys
from pathlib import Path

import experiment_v13_control as base

ORIGINAL_FRESH = base.fresh
ORIGINAL_ANSWERS = base.answers

# Proteger les jeux de validation et les jeux finaux.
held_values = set()
for name in (
    "instructions_v14/wording_validation.jsonl",
    "instructions_v15/field_selection.jsonl",
    "final_evaluation/general.jsonl",
    "final_evaluation/repetitions.jsonl",
):
    _, values = ORIGINAL_ANSWERS(Path(name))
    held_values.update(values)

def answers(path):
    records, values = ORIGINAL_ANSWERS(path)
    return records, values | held_values

def fresh(rng, used, reserved, maximum, task="copie", length=None):
    row = ORIGINAL_FRESH(
        rng, used, reserved, maximum, task, length
    )
    if task != "extraction":
        return row

    # Le generateur original fournit ces trois valeurs dans cet ordre.
    values = dict(zip(
        ("cible", "gauche", "droite"),
        row["source_values"],
    ))
    requested = rng.choice(("cible", "gauche", "droite"))
    fields = list(values.items())
    rng.shuffle(fields)
    text = " ; ".join(f"{key}={value}" for key, value in fields)

    templates = [
        f"Extrais uniquement la valeur du champ {requested} : {text}.",
        f"Donne seulement la valeur de {requested} dans : {text}.",
        f"Lis ces champs : {text}. Reponds uniquement avec la valeur de {requested}.",
        f"Quelle est la valeur de {requested} ? {text}. Sans commentaire.",
    ]
    answer = values[requested]
    row["messages"][0]["content"] = rng.choice(templates)
    row["messages"][1]["content"] = answer
    row["task"] = f"selection_{requested}_L{len(answer)}"

    # Enregistrer aussi les distracteurs devenus des reponses possibles.
    used.update(row["source_values"])
    return row

base.answers = answers
base.fresh = fresh

if __name__ == "__main__":
    # Le moteur enregistre une empreinte du programme d'experience.
    base.__file__ = __file__
    base.main()
