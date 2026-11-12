import json
import random
from collections import defaultdict
from pathlib import Path

from integrated_agent_v2 import Agent
from compare_adaptive_recovery import FIRST, interact
from history_reliability import upper_remaining_error_bound

THRESHOLDS = (0.1, 0.2, 0.4, 0.6)

def main():
    output = Path("autonomous_lab/runs/reliability_gated_recovery_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    initial = Agent()
    for _ in range(48):
        interact(initial, FIRST)
    assert len(initial.state["observations"]) == 32
    snapshot = json.dumps(initial.state)

    rng = random.Random(94101)
    pairs = rng.sample(
        [(a, b) for a in range(32) for b in range(a+1, 32)], 32
    )
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

    config = {
        "additional_questions": 16,
        "replacement_fit_questions": 8,
        "replacement_holdout_questions": 8,
        "alpha_per_bound": 0.025,
        "error_fraction_thresholds": list(THRESHOLDS),
        "selection": "lowest_accepted_upper_bound",
        "replacement_frozen_before_holdout": True,
        "noise": False,
        "stable_after_initial_change": True,
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    details = []
    for case_id, (group, changed) in enumerate(cases):
        hidden = tuple(
            value ^ int(index in changed)
            for index, value in enumerate(FIRST)
        )
        replacement = Agent(json.loads(snapshot))

        for _ in range(32):
            event = interact(replacement, hidden)
            if event["changed_answer_detected"]:
                break
        else:
            raise RuntimeError("Changement non detecte.")

        detected = event["input_index"]
        order = [i for i in range(32) if i != detected]
        random.Random(94101000+case_id).shuffle(order)
        fit, checks, remaining = order[:8], order[8:16], order[16:]

        for index in fit:
            interact(replacement, hidden, index=index)

        # Figer toutes les predictions avant de consulter les controles.
        proposed = [
            replacement.prediction(index)[0] for index in range(32)
        ]
        history_errors = sum(
            FIRST[index] != hidden[index] for index in order[:16]
        )
        # Une abstention compte comme erreur pour cette borne.
        replacement_errors = sum(
            proposed[index] != hidden[index] for index in checks
        )

        history_bound = upper_remaining_error_bound(
            31, 16, history_errors, alpha=0.025
        )
        replacement_bound = upper_remaining_error_bound(
            23, 8, replacement_errors, alpha=0.025
        )

        for threshold in THRESHOLDS:
            accepted = []
            if history_bound <= threshold:
                accepted.append(("history", history_bound))
            if replacement_bound <= threshold:
                accepted.append(("replacement", replacement_bound))

            selected = min(
                accepted, key=lambda item: item[1]
            )[0] if accepted else "abstain"

            answered = correct = 0
            for index in remaining:
                prediction = (
                    FIRST[index] if selected == "history"
                    else proposed[index] if selected == "replacement"
                    else None
                )
                if prediction is not None:
                    answered += 1
                    correct += prediction == hidden[index]

            details.append({
                "case_id": case_id,
                "group": group,
                "threshold": threshold,
                "history_bound": history_bound,
                "replacement_bound": replacement_bound,
                "history_control_errors": history_errors,
                "replacement_control_errors": replacement_errors,
                "selected": selected,
                "correct": correct,
                "answered": answered,
                "evaluated_inputs": len(remaining),
            })

    summaries = []
    groups = ("single_flip", "double_flip", "flip_4", "flip_8", "flip_16", "inversion")
    for threshold in THRESHOLDS:
        print(f"\nTOLERANCE : {100*threshold:.0f}%")
        for group in groups:
            rows = [
                row for row in details
                if row["threshold"] == threshold and row["group"] == group
            ]
            total = sum(row["evaluated_inputs"] for row in rows)
            answered = sum(row["answered"] for row in rows)
            correct = sum(row["correct"] for row in rows)
            selections = {
                name: sum(row["selected"] == name for row in rows)
                for name in ("history", "replacement", "abstain")
            }
            summaries.append({
                "threshold": threshold,
                "group": group,
                "correct": correct,
                "answered": answered,
                "evaluated_inputs": total,
                "selections": selections,
            })
            accuracy = f"{100*correct/answered:.1f}%" if answered else "N/A"
            print(
                f"{group} | precision={accuracy} | "
                f"couverture={answered}/{total} | choix={selections}"
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
