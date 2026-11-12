import itertools
import json
from collections import Counter
from pathlib import Path

import torch
import assistant_v04 as extension

app = extension.previous

@torch.no_grad()
def main():
    source = Path("instructions_v18/validation.jsonl")
    groups = {}
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            groups.setdefault(row["group_id"], row["source_values"])

    model = extension.factory(
        extension.core.load(Path("runs/experiment_v20_selection.pt"))
    ).eval()

    names = ("cible", "gauche", "droite")
    stats = Counter()
    details = []

    for group_id, values in sorted(groups.items()):
        # Le generateur v18 conserve les valeurs dans l'ordre des champs
        # affiches. Recuperer leurs noms depuis la question source.
        import re
        original = next(
            json.loads(line)
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and json.loads(line)["group_id"] == group_id
        )
        fields = dict(re.findall(
            r"(cible|gauche|droite)=([a-z0-9]+)",
            original["messages"][0]["content"],
        ))
        if set(fields) != set(names) or len(set(fields.values())) != 3:
            raise ValueError("Champs invalides.")

        for order in itertools.permutations(names):
            text = " ; ".join(f"{name}={fields[name]}" for name in order)

            for requested in names:
                question = (
                    f"Extrais uniquement la valeur du champ {requested} : "
                    f"{text}."
                )
                prefix, _ = app.conversation_prefix([], question, 512)
                result, ended = app.stream_bytes(
                    model, prefix, 96, 0, 40, 42, app.END.encode()
                )
                answer = result.decode("utf-8", errors="replace").strip()
                selected = next(
                    (name for name in names if answer == fields[name]),
                    "autre",
                )
                requested_position = order.index(requested) + 1
                selected_position = (
                    order.index(selected) + 1 if selected != "autre" else 0
                )

                stats[("demande", requested)] += 1
                stats[("choix", requested, selected)] += 1
                stats[("position_total", requested_position)] += 1
                stats[("position_choisie", selected_position)] += 1
                if answer == fields[requested] and ended:
                    stats[("correct", requested)] += 1
                    stats[("position_correct", requested_position)] += 1

                details.append({
                    "group_id": group_id,
                    "order": order,
                    "requested": requested,
                    "expected": fields[requested],
                    "answer": answer,
                    "ended": ended,
                    "selected_field": selected,
                })

    output = Path("v21_position_details.jsonl")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
        encoding="utf-8",
    )

    print("MODELE : experiment_v20_selection.pt")
    print(f"QUESTIONS : {len(details)}")
    for requested in names:
        print(
            f"DEMANDE {requested} | "
            f"exact+EOS={stats[('correct', requested)]}/"
            f"{stats[('demande', requested)]} | "
            + " | ".join(
                f"repond_{selected}={stats[('choix', requested, selected)]}"
                for selected in (*names, "autre")
            )
        )
    for position in (1, 2, 3):
        print(
            f"CHAMP_DEMANDE_POSITION {position} | "
            f"exact+EOS={stats[('position_correct', position)]}/"
            f"{stats[('position_total', position)]}"
        )
    print(
        "POSITION_DE_LA_VALEUR_REPONDUE | "
        + " | ".join(
            f"{position}={stats[('position_choisie', position)]}"
            for position in (1, 2, 3)
        )
        + f" | autre={stats[('position_choisie', 0)]}"
    )
    print(f"Details : {output}")

if __name__ == "__main__":
    main()
