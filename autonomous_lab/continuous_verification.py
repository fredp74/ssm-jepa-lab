import json
import random
from collections import Counter
from pathlib import Path

def simulate(proposal, observed, ask, policy, seed):
    rng = random.Random(seed)
    checked = set(observed)

    for step in range(1, 65):
        remaining = [i for i in range(32) if i not in checked]
        if not remaining:
            return {
                "status": "fully_verified",
                "additional_questions": step-1,
            }

        index = (
            rng.choice(remaining)
            if policy == "unseen_audit"
            else rng.randrange(32)
        )
        response = ask(index)
        checked.add(index)

        if response != proposal[index]:
            return {
                "status": "revoked",
                "additional_questions": step,
                "counterexample_index": index,
            }

    return {
        "status": "fully_verified" if len(checked) == 32 else "provisional",
        "additional_questions": 64,
    }

def main():
    source = Path(
        "autonomous_lab/runs/budgeted_five_bits_01/results.json"
    )
    output = Path("autonomous_lab/runs/continuous_verification_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    previous = json.loads(source.read_text(encoding="utf-8"))
    accepted = [
        row for row in previous["details"]
        if row["status"] == "provisional"
    ]
    output.mkdir(parents=True)
    details = []

    for case_id, row in enumerate(accepted):
        proposal = tuple(map(int, row["proposal"]))
        hidden = tuple(map(int, row["hidden_table"]))
        observed = {
            item["input_index"]: item["answer"]
            for item in row["discovery_observations"]
        }
        observed.update({
            item["input_index"]: item["answer"]
            for item in row["verification"]
        })
        assert all(proposal[i] == answer for i, answer in observed.items())

        def ask(index):
            return hidden[index]

        for repetition in range(5):
            seed = 82000000 + case_id*10 + repetition
            for policy in ("random_usage", "unseen_audit"):
                result = simulate(proposal, observed, ask, policy, seed)
                details.append({
                    "case_id": case_id,
                    "group": row["group"],
                    "repetition": repetition,
                    "policy": policy,
                    "initial_rule_wrong": proposal != hidden,
                    "initial_checked_inputs": len(observed),
                    **result,
                })

    summaries = []
    for group in ("known", "near_miss", "random_outside"):
        for policy in ("random_usage", "unseen_audit"):
            rows = [
                row for row in details
                if row["group"] == group and row["policy"] == policy
            ]
            counts = Counter(row["status"] for row in rows)
            delays = [
                row["additional_questions"] for row in rows
                if row["status"] == "revoked"
            ]
            detected_by = {
                str(budget): sum(
                    row["status"] == "revoked"
                    and row["additional_questions"] <= budget
                    for row in rows
                )
                for budget in (1, 4, 8, 16, 32, 64)
            }
            summary = {
                "group": group,
                "policy": policy,
                "trials": len(rows),
                "statuses": dict(counts),
                "mean_questions_until_stop": (
                    sum(row["additional_questions"] for row in rows)/len(rows)
                ),
                "mean_detection_delay_among_revoked": (
                    sum(delays)/len(delays) if delays else None
                ),
                "detected_by_budget": detected_by,
            }
            summaries.append(summary)
            print(f"\n{group} | {policy} | essais={len(rows)}")
            print(f"Statuts : {dict(counts)}")
            print(f"Detectees par budget : {detected_by}")
            if delays:
                print(f"Delai moyen des revocations : {sum(delays)/len(delays):.2f}")

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "source": str(source),
                "accepted_functions": len(accepted),
                "repetitions_per_function": 5,
                "maximum_additional_questions": 64,
                "revocation": "first_observed_counterexample",
                "repair_after_revocation": False,
                "noise": False,
            },
            "summaries": summaries,
            "details": details,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
