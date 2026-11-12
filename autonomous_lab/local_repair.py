import json
import random
from collections import Counter
from pathlib import Path

from budgeted_five_bits import FAMILY

def repair_family():
    rules = set(FAMILY)
    for base in FAMILY:
        for index in range(32):
            changed = list(base)
            changed[index] ^= 1
            rules.add(tuple(changed))
    return sorted(rules)

def compatible(rule, observations):
    return all(rule[index] == answer for index, answer in observations.items())

def choose(candidates, observations):
    available = [i for i in range(32) if i not in observations]
    def score(index):
        ones = sum(rule[index] for rule in candidates)
        return min(ones, len(candidates)-ones)
    return max(available, key=score)

def main():
    source = Path("autonomous_lab/runs/budgeted_five_bits_01/results.json")
    output = Path("autonomous_lab/runs/local_repair_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    previous = json.loads(source.read_text(encoding="utf-8"))
    cases = [r for r in previous["details"] if r["status"] == "provisional"]
    expanded = repair_family()
    output.mkdir(parents=True)

    config = {
        "source": str(source),
        "repair_family": "known_rule_plus_at_most_one_flipped_output",
        "repair_family_size": len(expanded),
        "maximum_total_queried_inputs": 24,
        "maximum_monitoring_questions": 8,
        "required_final_checks": 4,
        "minimum_evaluator_only_inputs": 8,
        "noise": False,
        "seed": 83001,
    }
    (output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    results = []
    for case_id, row in enumerate(cases):
        hidden = tuple(map(int, row["hidden_table"]))
        original = tuple(map(int, row["proposal"]))
        observations = {
            item["input_index"]: item["answer"]
            for item in row["discovery_observations"]
        }
        observations.update({
            item["input_index"]: item["answer"]
            for item in row["verification"]
        })
        initial_evidence = dict(observations)
        rng = random.Random(83001+case_id)
        counterexample = None

        def ask(index):
            if index in observations:
                raise RuntimeError("Entree deja interrogee.")
            observations[index] = hidden[index]
            return observations[index]

        # Surveillance de la proposition initiale.
        for _ in range(8):
            if len(observations) >= 20:
                break
            available = [i for i in range(32) if i not in observations]
            index = rng.choice(available)
            answer = ask(index)
            if answer != original[index]:
                counterexample = {"input_index": index, "answer": answer}
                break

        proposal = original
        status = "unchanged_provisional"
        checks = 0

        if counterexample is not None:
            proposal = None
            candidates = [
                rule for rule in expanded
                if compatible(rule, observations)
            ]

            # Reserver quatre entrees du budget a la verification.
            while len(candidates) > 1 and len(observations) < 20:
                index = choose(candidates, observations)
                answer = ask(index)
                candidates = [
                    rule for rule in candidates if rule[index] == answer
                ]

            if not candidates:
                status = "repair_family_contradicted"
            elif len(candidates) > 1:
                status = "repair_abstention"
            else:
                proposal = candidates[0]
                assert compatible(proposal, observations)
                status = "repair_provisional"

                # Proposition figee avant ces quatre controles.
                available = [i for i in range(32) if i not in observations]
                for index in rng.sample(available, 4):
                    answer = ask(index)
                    checks += 1
                    if answer != proposal[index]:
                        status = "repair_revoked"
                        break

        assert len(observations) <= 24
        assert all(observations[i] == y for i, y in initial_evidence.items())

        accepted = status in ("unchanged_provisional", "repair_provisional")
        remaining = [i for i in range(32) if i not in observations]
        assert len(remaining) >= 8

        results.append({
            "group": row["group"],
            "status": status,
            "counterexample": counterexample,
            "initial_observations": len(initial_evidence),
            "total_questions": len(observations),
            "final_checks": checks,
            "false_acceptance": accepted and proposal != hidden,
            "correct_unqueried": (
                sum(proposal[i] == hidden[i] for i in remaining)
                if accepted else 0
            ),
            "evaluated_unqueried": len(remaining) if accepted else 0,
            "original_proposal": "".join(map(str, original)),
            "final_proposal": (
                "".join(map(str, proposal)) if proposal is not None else None
            ),
            "observations": [
                {"input_index": i, "answer": y}
                for i, y in observations.items()
            ],
        })

    summaries = {}
    for group in ("known", "near_miss", "random_outside"):
        rows = [r for r in results if r["group"] == group]
        summary = {
            "functions": len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "false_acceptances": sum(r["false_acceptance"] for r in rows),
            "questions": sum(r["total_questions"] for r in rows),
            "correct_unqueried": sum(r["correct_unqueried"] for r in rows),
            "evaluated_unqueried": sum(r["evaluated_unqueried"] for r in rows),
        }
        summaries[group] = summary
        print(f"{group} : {summary}", flush=True)

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": config,
            "summaries": summaries,
            "details": results,
        }, indent=2), encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
