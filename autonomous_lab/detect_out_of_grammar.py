import itertools
import json
from collections import Counter
from pathlib import Path

from consolidation import Learner, RULES, INPUTS

def run(hidden, audit):
    learner = Learner()
    candidates = list(RULES)
    asked = set()
    observations = []
    provisional = None

    while True:
        if not candidates:
            return {
                "status": "incompatible",
                "questions": len(asked),
                "provisional_rule": provisional,
                "observations": observations,
            }

        if len(candidates) == 1:
            inferred = candidates[0]
            if provisional is None:
                provisional = "".join(map(str, inferred))

            if not audit or len(asked) == len(INPUTS):
                return {
                    "status": (
                        "accepted_correct" if inferred == hidden
                        else "accepted_wrong"
                    ),
                    "questions": len(asked),
                    "provisional_rule": provisional,
                    "observations": observations,
                }

            # Une seule hypothese reste : interroger une entree
            # non encore testee. Aucun acces a la reponse cachee
            # n'est utilise pour choisir la question.
            index = next(i for i in range(8) if i not in asked)
        else:
            index = learner.choose(candidates, asked)

        answer = hidden[index]
        asked.add(index)
        observations.append({
            "input": list(INPUTS[index]),
            "output": answer,
        })
        candidates = [
            rule for rule in candidates if rule[index] == answer
        ]

def main():
    output = Path("autonomous_lab/runs/out_of_grammar_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    known = set(RULES)
    details = []
    summaries = {}

    for policy, audit in (("early_stop", False), ("full_audit", True)):
        groups = {
            "in_grammar": Counter(),
            "out_of_grammar": Counter(),
        }

        for hidden in itertools.product((0, 1), repeat=8):
            group = "in_grammar" if hidden in known else "out_of_grammar"
            result = run(hidden, audit)

            groups[group]["functions"] += 1
            groups[group][result["status"]] += 1
            groups[group]["questions"] += result["questions"]
            details.append({
                "policy": policy,
                "group": group,
                "hidden_table": "".join(map(str, hidden)),
                **result,
            })

        summaries[policy] = {
            group: dict(counts) for group, counts in groups.items()
        }

        print(f"\nPOLITIQUE : {policy}")
        for group, counts in groups.items():
            print(
                f"{group} | fonctions={counts['functions']} | "
                f"incompatibles={counts['incompatible']} | "
                f"acceptees_correctes={counts['accepted_correct']} | "
                f"acceptees_fausses={counts['accepted_wrong']} | "
                f"questions_moyennes="
                f"{counts['questions']/counts['functions']:.3f}"
            )

    assert summaries["full_audit"]["out_of_grammar"].get(
        "accepted_wrong", 0
    ) == 0
    assert summaries["full_audit"]["in_grammar"].get(
        "accepted_correct", 0
    ) == len(RULES)

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "input_bits": 3,
                "all_functions": 256,
                "known_rules": len(RULES),
                "library": "empty",
                "noise": False,
                "full_audit": "query_until_contradiction_or_all_inputs",
            },
            "summaries": summaries,
            "details": details,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
