import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path

SIZE = 32
TERMS = [mask for mask in range(SIZE) if mask.bit_count() <= 2]

def table(terms):
    return tuple(
        sum((x & mask) == mask for mask in terms) % 2
        for x in range(SIZE)
    )

FAMILY = sorted({
    table(terms)
    for count in (0, 1, 2)
    for terms in itertools.combinations(TERMS, count)
})
assert len(FAMILY) == 137

def signature(rule):
    return "".join(map(str, rule))

def make_cases():
    rng = random.Random(81001)
    known = set(FAMILY)
    used = set(known)
    cases = [("known", rule) for rule in FAMILY]

    while sum(group == "near_miss" for group, _ in cases) < 100:
        values = list(rng.choice(FAMILY))
        index = rng.randrange(SIZE)
        values[index] ^= 1
        rule = tuple(values)
        if rule not in used:
            used.add(rule)
            cases.append(("near_miss", rule))

    while sum(group == "random_outside" for group, _ in cases) < 100:
        rule = tuple(rng.randrange(2) for _ in range(SIZE))
        if rule not in used:
            used.add(rule)
            cases.append(("random_outside", rule))
    return cases

def acquire(ask):
    candidates = list(FAMILY)
    observed = {}

    for _ in range(12):
        if len(candidates) <= 1:
            break
        available = [i for i in range(SIZE) if i not in observed]

        def score(index):
            ones = sum(rule[index] for rule in candidates)
            return min(ones, len(candidates)-ones)

        index = max(available, key=score)
        response = ask(index)
        observed[index] = response
        candidates = [
            rule for rule in candidates if rule[index] == response
        ]

    if not candidates:
        return None, observed, "contradiction"
    if len(candidates) > 1:
        return None, observed, "abstention"
    return candidates[0], observed, "proposal"

def main():
    output = Path("autonomous_lab/runs/budgeted_five_bits_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    cases = make_cases()
    protocol = {
        "input_bits": 5,
        "domain_size": SIZE,
        "family_size": len(FAMILY),
        "family": "XOR of at most two degree-at-most-two monomials",
        "discovery_budget": 12,
        "verification_budget": 4,
        "verification": "random_unqueried_inputs",
        "proposal_frozen_before_verification": True,
        "seed": 81001,
        "noise": False,
    }
    # Enregistrer les cas avant leur evaluation.
    (output / "protocol.json").write_text(
        json.dumps({
            **protocol,
            "cases": [
                {"group": group, "table": signature(rule)}
                for group, rule in cases
            ],
        }, indent=2), encoding="utf-8",
    )

    details = []
    summaries = {}

    for group, hidden in cases:
        queried = {}

        def ask(index):
            if index in queried:
                raise RuntimeError("Question repetee.")
            queried[index] = hidden[index]
            return hidden[index]

        proposal, discovery, status = acquire(ask)
        discovery_count = len(queried)
        verification = []

        if proposal is not None:
            seed = int(hashlib.sha256(
                signature(hidden).encode()
            ).hexdigest()[:16], 16)
            rng = random.Random(seed)
            available = [i for i in range(SIZE) if i not in queried]
            checks = rng.sample(available, 4)

            status = "provisional"
            for index in checks:
                response = ask(index)
                verification.append({
                    "input_index": index,
                    "answer": response,
                    "predicted": proposal[index],
                })
                if response != proposal[index]:
                    status = "contradiction"
                    break

        assert discovery_count <= 12
        assert len(verification) <= 4
        assert len(queried) <= 16

        # Evaluation uniquement : ces reponses ne sont jamais
        # communiquees a l'agent.
        remaining = [i for i in range(SIZE) if i not in queried]
        accepted = status == "provisional"
        correct = (
            sum(proposal[i] == hidden[i] for i in remaining)
            if accepted else 0
        )
        false_acceptance = accepted and proposal != hidden

        details.append({
            "group": group,
            "hidden_table": signature(hidden),
            "status": status,
            "proposal": signature(proposal) if proposal is not None else None,
            "discovery_questions": discovery_count,
            "verification_questions": len(verification),
            "verification": verification,
            "questions_total": len(queried),
            "false_acceptance": false_acceptance,
            "evaluated_unqueried": len(remaining) if accepted else 0,
            "correct_unqueried": correct,
            "discovery_observations": [
                {"input_index": i, "answer": answer}
                for i, answer in discovery.items()
            ],
        })

    for group in ("known", "near_miss", "random_outside"):
        rows = [row for row in details if row["group"] == group]
        counts = Counter(row["status"] for row in rows)
        tested = sum(row["evaluated_unqueried"] for row in rows)
        correct = sum(row["correct_unqueried"] for row in rows)
        summary = {
            "functions": len(rows),
            "statuses": dict(counts),
            "false_acceptances": sum(row["false_acceptance"] for row in rows),
            "questions": sum(row["questions_total"] for row in rows),
            "correct_unqueried": correct,
            "evaluated_unqueried": tested,
        }
        summaries[group] = summary
        accuracy = f"{100*correct/tested:.2f} %" if tested else "non applicable"
        print(
            f"{group} | fonctions={len(rows)} | statuts={dict(counts)} | "
            f"fausses_acceptations={summary['false_acceptances']} | "
            f"questions_moyennes={summary['questions']/len(rows):.2f} | "
            f"precision_sur_acceptations={accuracy}",
            flush=True,
        )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": protocol,
            "summaries": summaries,
            "details": details,
        }, indent=2), encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
