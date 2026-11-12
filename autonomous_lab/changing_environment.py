import json
import random
from pathlib import Path

from consolidation import Learner, RULES

def main():
    output = Path("autonomous_lab/runs/changing_environment_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    episodes = []
    scenarios = []

    for seed in (72001, 72002, 72003, 72004, 72005):
        rng = random.Random(seed)
        frequent = rng.sample(range(len(RULES)), 6)
        first, second = frequent[:3], frequent[3:]
        sequence = []

        for episode in range(60):
            hot = first if episode < 30 else second
            rule_id = (
                rng.choice(hot) if rng.random() < 0.8
                else rng.randrange(len(RULES))
            )
            sequence.append(rule_id)

        scenarios.append({
            "seed": seed,
            "first_frequent_rules": first,
            "second_frequent_rules": second,
            "sequence": sequence,
        })
        cumulative = Learner()
        discounted = Learner()

        for number, rule_id in enumerate(sequence, 1):
            hidden = RULES[rule_id]

            def ask(index):
                return hidden[index]

            # Oubli continu : aucun signal de changement de phase.
            discounted.counts = {
                key: value*0.95
                for key, value in discounted.counts.items()
            }

            for strategy, agent in (
                ("cold", Learner()),
                ("cumulative", cumulative),
                ("discounted", discounted),
            ):
                inferred, observations, asked, known = agent.acquire(ask)
                if inferred != hidden:
                    raise RuntimeError("Regle identifiee incorrecte.")

                episodes.append({
                    "seed": seed,
                    "episode": number,
                    "phase": 1 if number <= 30 else 2,
                    "strategy": strategy,
                    "questions": len(asked),
                    "previously_known": known,
                    "inferred_rule": "".join(map(str, inferred)),
                    "observations": observations,
                    "correct_unqueried": sum(
                        inferred[i] == hidden[i]
                        for i in range(8) if i not in asked
                    ),
                    "total_unqueried": 8-len(asked),
                })

        for label, agent in (
            ("cumulative", cumulative),
            ("discounted", discounted),
        ):
            path = output / f"{label}_{seed}_library.json"
            agent.save(path)
            restored = Learner.load(path)
            assert restored.counts == agent.counts

        print(f"Sequence {seed} terminee.", flush=True)

    summaries = []
    for label, start, end in (
        ("phase_1", 1, 30),
        ("debut_phase_2", 31, 40),
        ("fin_phase_2", 41, 60),
        ("total", 1, 60),
    ):
        print(f"\n{label}")
        for strategy in ("cold", "cumulative", "discounted"):
            rows = [
                row for row in episodes
                if row["strategy"] == strategy
                and start <= row["episode"] <= end
            ]
            questions = sum(row["questions"] for row in rows)
            summary = {
                "period": label,
                "strategy": strategy,
                "episodes": len(rows),
                "questions": questions,
                "mean_questions": questions/len(rows),
                "correct_unqueried": sum(
                    row["correct_unqueried"] for row in rows
                ),
                "total_unqueried": sum(
                    row["total_unqueried"] for row in rows
                ),
            }
            summaries.append(summary)
            print(
                f"{strategy} | questions={questions} | "
                f"moyenne={questions/len(rows):.3f}"
            )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "seeds": [72001, 72002, 72003, 72004, 72005],
                "episodes_per_sequence": 60,
                "switch_after_episode": 30,
                "switch_announced_to_agent": False,
                "frequent_rule_probability": 0.8,
                "discount_per_episode": 0.95,
                "grammar_size": len(RULES),
                "stop_condition": "unique_compatible_rule",
            },
            "scenarios": scenarios,
            "summaries": summaries,
            "episodes": episodes,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
