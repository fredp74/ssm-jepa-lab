import json
from pathlib import Path

from integrated_agent import Agent as OriginalAgent
from integrated_agent_v2 import Agent as CorrectedAgent

FIRST = tuple(
    ((index >> 4) & 1) ^ int(index == 17)
    for index in range(32)
)

def interact(agent, hidden):
    index, purpose = agent.choose_question()
    prediction, kind = agent.prediction(index)
    return agent.observe(
        index, hidden[index], prediction, kind, purpose
    )

def prepare(agent_class):
    agent = agent_class()
    for _ in range(48):
        interact(agent, FIRST)

    if len(agent.state["observations"]) != 32:
        raise RuntimeError("Les 32 entrees doivent etre memorisees.")
    if any(
        agent.state["observations"][str(i)] != FIRST[i]
        for i in range(32)
    ):
        raise RuntimeError("Memoire initiale incorrecte.")
    return json.dumps(agent.state)

def main():
    output = Path("autonomous_lab/runs/audit_coverage_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    details = []
    summaries = []

    for label, agent_class in (
        ("original", OriginalAgent),
        ("corrected", CorrectedAgent),
    ):
        snapshot = prepare(agent_class)

        for changed_index in range(32):
            agent = agent_class(json.loads(snapshot))
            hidden = list(FIRST)
            hidden[changed_index] ^= 1
            delay = None
            audit_inputs = []

            for interaction in range(1, 129):
                event = interact(agent, hidden)
                audit_inputs.append(event["input_index"])

                if event["changed_answer_detected"]:
                    if event["input_index"] != changed_index:
                        raise RuntimeError("Detection sur une mauvaise entree.")
                    delay = interaction
                    break

            details.append({
                "policy": label,
                "changed_index": changed_index,
                "detection_delay": delay,
                "queried_inputs": audit_inputs,
            })

        rows = [row for row in details if row["policy"] == label]
        delays = [
            row["detection_delay"] for row in rows
            if row["detection_delay"] is not None
        ]
        summary = {
            "policy": label,
            "changes": len(rows),
            "detected": len(delays),
            "mean_delay_among_detected": sum(delays)/len(delays) if delays else None,
            "minimum_delay": min(delays) if delays else None,
            "maximum_delay": max(delays) if delays else None,
            "detected_by_budget": {
                str(budget): sum(
                    row["detection_delay"] is not None
                    and row["detection_delay"] <= budget
                    for row in rows
                )
                for budget in (4, 8, 16, 32, 64, 128)
            },
        }
        summaries.append(summary)
        print(f"\n{label}")
        print(json.dumps(summary, indent=2))

        if label == "corrected":
            assert len(delays) == 32
            assert max(delays) <= 32

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "warmup_interactions": 48,
                "initial_domain_fully_observed": True,
                "changed_inputs_per_trial": 1,
                "all_change_positions_tested": True,
                "maximum_post_change_interactions": 128,
                "stop_on_first_detection": True,
                "noise": False,
                "neural_model_used": False,
            },
            "summaries": summaries,
            "details": details,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
