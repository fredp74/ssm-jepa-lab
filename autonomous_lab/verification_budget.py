import itertools
import json
from collections import Counter
from pathlib import Path

from consolidation import Learner, RULES

def run(hidden, extra):
    learner = Learner()
    candidates = list(RULES)
    asked = set()
    verification_questions = 0
    provisional = False

    while True:
        if not candidates:
            return "rejected", len(asked), verification_questions

        if len(candidates) == 1:
            provisional = True
            if len(asked) == 8:
                return "verified", len(asked), verification_questions
            if extra is not None and verification_questions >= extra:
                return "provisional", len(asked), verification_questions

            index = next(i for i in range(8) if i not in asked)
        else:
            index = learner.choose(candidates, asked)

        answer = hidden[index]
        asked.add(index)
        if provisional:
            verification_questions += 1
        candidates = [
            rule for rule in candidates if rule[index] == answer
        ]

def main():
    output = Path("autonomous_lab/runs/verification_budget_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    known = set(RULES)
    summaries = []
    details = []

    for label, extra in (
        ("extra_0", 0),
        ("extra_1", 1),
        ("extra_2", 2),
        ("extra_3", 3),
        ("full_audit", None),
    ):
        counters = {
            "in_grammar": Counter(),
            "out_of_grammar": Counter(),
        }

        for hidden in itertools.product((0, 1), repeat=8):
            group = "in_grammar" if hidden in known else "out_of_grammar"
            status, questions, checks = run(hidden, extra)
            counts = counters[group]
            counts["functions"] += 1
            counts["questions"] += questions
            counts[status] += 1

            if group == "out_of_grammar" and status != "rejected":
                counts["false_acceptances"] += 1

            details.append({
                "policy": label,
                "group": group,
                "hidden_table": "".join(map(str, hidden)),
                "status": status,
                "questions": questions,
                "verification_questions": checks,
            })

        print(f"\n{label}")
        for group, counts in counters.items():
            summaries.append({
                "policy": label,
                "group": group,
                **dict(counts),
            })
            print(
                f"{group} | questions_moyennes="
                f"{counts['questions']/counts['functions']:.3f} | "
                f"rejets={counts['rejected']} | "
                f"acceptations_provisoires={counts['provisional']} | "
                f"verification_complete={counts['verified']} | "
                f"fausses_acceptations={counts['false_acceptances']}"
            )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "functions": 256,
                "known_rules": len(RULES),
                "verification_order": "ascending_unqueried_input_index",
                "verification_budgets": [0, 1, 2, 3, "all_remaining"],
                "noise": False,
                "library": "empty",
            },
            "summaries": summaries,
            "details": details,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
