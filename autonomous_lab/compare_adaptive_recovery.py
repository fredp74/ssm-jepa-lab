import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from integrated_agent_v2 import Agent

FIRST = tuple(
    ((index >> 4) & 1) ^ int(index == 17)
    for index in range(32)
)
BUDGETS = (0, 4, 8, 16, 31)

def interact(agent, hidden, index=None):
    if index is None:
        index, purpose = agent.choose_question()
    else:
        purpose = "shared_reverification"
    prediction, kind = agent.prediction(index)
    event = agent.observe(
        index, hidden[index], prediction, kind, purpose
    )
    return event

def main():
    output = Path("autonomous_lab/runs/adaptive_recovery_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    initial = Agent()
    for _ in range(48):
        interact(initial, FIRST)
    assert len(initial.state["observations"]) == 32
    snapshot = json.dumps(initial.state)

    rng = random.Random(94101)
    pairs = rng.sample(list(itertools.combinations(range(32), 2)), 32)
    cases = (
        [("single_flip", (i,)) for i in range(32)]
        + [("double_flip", pair) for pair in pairs]
        + [
            (f"flip_{size}", tuple(sorted(rng.sample(range(32), size))))
            for size in (4, 8, 16)
            for _ in range(32)
        ]
        + [("inversion", tuple(range(32)))]
    )

    output.mkdir(parents=True)
    config = {
        "single_flip_cases": 32,
        "double_flip_cases": 32,
        "inversion_cases": 1,
        "additional_budgets_after_detection": list(BUDGETS),
        "same_post_detection_observations_for_both_methods": True,
        "historical_values_are_provisional": True,
        "case_type_disclosed_to_agent": False,
        "seed": 94101,
        "intermediate_flip_counts": [4, 8, 16],
        "sampled_cases_per_intermediate_count": 32,
        "adaptive_minimum_checks": 4,
        "adaptive_reset_threshold": 0.5,
        "adaptive_early_behavior": "abstain_on_unreobserved_inputs",
        "adaptive_policy_is_handwritten": True,
        "noise": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    details = []
    for case_id, (group, changed) in enumerate(cases):
        hidden = tuple(
            value ^ int(index in changed)
            for index, value in enumerate(FIRST)
        )
        reset = Agent(json.loads(snapshot))

        for delay in range(1, 33):
            event = interact(reset, hidden)
            if event["changed_answer_detected"]:
                break
        else:
            raise RuntimeError("Changement non detecte.")

        detected_index = event["input_index"]
        # La reinitialisation est deja effectuee par Agent.observe.
        assert len(reset.state["observations"]) == 1

        historical = list(FIRST)
        historical[detected_index] = event["answer"]
        fresh = {detected_index}

        order = [i for i in range(32) if i != detected_index]
        random.Random(94101000+case_id).shuffle(order)

        for budget in range(32):
            if budget:
                index = order[budget-1]
                interact(reset, hidden, index=index)
                historical[index] = hidden[index]
                fresh.add(index)

            if budget not in BUDGETS:
                continue

            remaining = [i for i in range(32) if i not in fresh]
            for method in ("reset", "local_history", "adaptive"):
                answered = correct = 0
                for index in remaining:
                    if method == "reset":
                        prediction = reset.prediction(index)[0]
                    elif method == "local_history":
                        prediction = historical[index]
                    else:
                        # Ne pas compter la contradiction ayant declenche
                        # la detection comme un controle aleatoire.
                        checked = [i for i in fresh if i != detected_index]
                        mismatches = sum(
                            historical[i] != FIRST[i] for i in checked
                        )
                        if len(checked) < 4:
                            prediction = None
                        elif 2*mismatches >= len(checked):
                            prediction = reset.prediction(index)[0]
                        else:
                            prediction = historical[index]
                    if prediction is not None:
                        answered += 1
                        correct += prediction == hidden[index]

                details.append({
                    "case_id": case_id,
                    "group": group,
                    "method": method,
                    "additional_questions": budget,
                    "detection_delay": delay,
                    "fresh_inputs": len(fresh),
                    "unqueried_current_inputs": len(remaining),
                    "answered": answered,
                    "correct": correct,
                })

    summaries = []
    for group in ("single_flip", "double_flip", "flip_4", "flip_8", "flip_16", "inversion"):
        print(f"\n{group}")
        for budget in BUDGETS:
            if budget == 31:
                continue
            for method in ("reset", "local_history", "adaptive"):
                rows = [
                    row for row in details
                    if row["group"] == group
                    and row["additional_questions"] == budget
                    and row["method"] == method
                ]
                total = sum(row["unqueried_current_inputs"] for row in rows)
                answered = sum(row["answered"] for row in rows)
                correct = sum(row["correct"] for row in rows)
                summary = {
                    "group": group,
                    "method": method,
                    "additional_questions": budget,
                    "cases": len(rows),
                    "correct": correct,
                    "answered": answered,
                    "unqueried_current_inputs": total,
                }
                summaries.append(summary)
                accuracy = f"{100*correct/answered:.1f}%" if answered else "N/A"
                print(
                    f"budget={budget} | {method} | "
                    f"precision={accuracy} | couverture={answered}/{total}"
                )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": config,
            "summaries": summaries,
            "details": details,
        }, indent=2), encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
