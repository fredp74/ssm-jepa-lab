import itertools
import json
from pathlib import Path

from baseline import Agent, INPUTS, RULES, predict

def trial(rule, questions):
    agent = Agent()
    for question in questions:
        agent.observe(question, predict(rule, question))

    asked = set(questions)
    heldout = [x for x in INPUTS if x not in asked]
    correct = sum(
        agent.answer(x) == predict(rule, x)
        for x in heldout
    )
    return {
        "correct": correct,
        "tested": len(heldout),
        "hypotheses": len(agent.hypotheses()),
    }

def main():
    output = Path("autonomous_lab/runs/exploration_comparison")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    results = {"active": [], "random": []}
    question_sets = list(itertools.combinations(INPUTS, 4))

    for rule_id, rule in enumerate(RULES):
        # L'agent choisit sans recevoir la regle cachee.
        agent = Agent()
        questions = []
        for _ in range(4):
            question = agent.choose_question()
            questions.append(question)
            agent.observe(question, predict(rule, question))

        active = trial(rule, questions)
        active["rule_id"] = rule_id
        results["active"].append(active)

        for questions in question_sets:
            random_result = trial(rule, questions)
            random_result["rule_id"] = rule_id
            results["random"].append(random_result)

    print(f"REGLES : {len(RULES)}")
    print(f"ENSEMBLES DE QUATRE QUESTIONS : {len(question_sets)}")

    for strategy, rows in results.items():
        correct = sum(row["correct"] for row in rows)
        tested = sum(row["tested"] for row in rows)
        identified = sum(row["hypotheses"] == 1 for row in rows)
        perfect = sum(row["correct"] == row["tested"] for row in rows)
        mean_hypotheses = sum(row["hypotheses"] for row in rows) / len(rows)

        print(f"\nSTRATEGIE : {strategy}")
        print(f"Essais : {len(rows)}")
        print(f"Predictions exactes : {correct}/{tested} ({100*correct/tested:.2f} %)")
        print(f"Regle identifiee : {identified}/{len(rows)}")
        print(f"Essais entierement corrects : {perfect}/{len(rows)}")
        print(f"Hypotheses restantes en moyenne : {mean_hypotheses:.3f}")

    (output / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {output}")

if __name__ == "__main__":
    main()
