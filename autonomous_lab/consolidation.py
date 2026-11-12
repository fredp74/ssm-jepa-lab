import itertools
import json
import math
import random
from pathlib import Path

INPUTS = list(itertools.product((0, 1), repeat=3))

def grammar():
    tables = set()
    for coefficients in itertools.product((0, 1), repeat=4):
        tables.add(tuple(
            (coefficients[0] + sum(
                a*b for a, b in zip(coefficients[1:], x)
            )) % 2
            for x in INPUTS
        ))
    for a, b in itertools.combinations(range(3), 2):
        for operation in ("and", "or"):
            for complement in (0, 1):
                tables.add(tuple(
                    (
                        (x[a] & x[b]) if operation == "and"
                        else (x[a] | x[b])
                    ) ^ complement
                    for x in INPUTS
                ))
    return sorted(tables)

RULES = grammar()

def signature(rule):
    return "".join(map(str, rule))

class Learner:
    def __init__(self, counts=None):
        self.counts = dict(counts or {})

    def choose(self, candidates, asked):
        weights = [
            1 + self.counts.get(signature(rule), 0)
            for rule in candidates
        ]
        total = sum(weights)

        def entropy(index):
            p = sum(
                weight * rule[index]
                for rule, weight in zip(candidates, weights)
            ) / total
            if p == 0 or p == 1:
                return 0.0
            return -p*math.log2(p)-(1-p)*math.log2(1-p)

        available = [i for i in range(8) if i not in asked]
        return max(available, key=entropy)

    def acquire(self, ask_environment):
        candidates = list(RULES)
        observations = []
        asked = set()

        while len(candidates) > 1:
            index = self.choose(candidates, asked)
            answer = ask_environment(index)
            asked.add(index)
            observations.append({"input": list(INPUTS[index]), "output": answer})
            candidates = [
                rule for rule in candidates if rule[index] == answer
            ]
            if not candidates:
                raise ValueError("Regle hors de la grammaire connue.")

        rule = candidates[0]
        key = signature(rule)
        previously_known = key in self.counts
        self.counts[key] = self.counts.get(key, 0) + 1
        return rule, observations, asked, previously_known

    def save(self, path):
        path.write_text(
            json.dumps({
                "format": 1,
                "grammar": [signature(rule) for rule in RULES],
                "counts": self.counts,
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data["format"] != 1
            or data["grammar"] != [signature(rule) for rule in RULES]
        ):
            raise ValueError("Bibliotheque incompatible.")
        return cls(data["counts"])

def main():
    output = Path("autonomous_lab/runs/consolidation_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    rng = random.Random(71001)
    frequent = rng.sample(range(len(RULES)), 3)
    sequence = [
        rng.choice(frequent) if rng.random() < 0.8
        else rng.randrange(len(RULES))
        for _ in range(60)
    ]

    persistent = Learner()
    library_path = output / "library.json"
    results = []

    for episode, rule_id in enumerate(sequence, 1):
        hidden = RULES[rule_id]

        # L'agent ne recoit que les reponses a ses questions.
        def ask(index):
            return hidden[index]

        row = {"episode": episode}
        for label, learner in (
            ("cold", Learner()),
            ("persistent", persistent),
        ):
            inferred, observations, asked, known = learner.acquire(ask)
            remaining = [i for i in range(8) if i not in asked]
            correct = sum(inferred[i] == hidden[i] for i in remaining)
            row[label] = {
                "questions": len(asked),
                "observations": observations,
                "previously_known": known,
                "correct_unqueried": correct,
                "total_unqueried": len(remaining),
                "inferred_rule": signature(inferred),
            }
            assert inferred == hidden

        persistent.save(library_path)
        restored = Learner.load(library_path)
        assert restored.counts == persistent.counts
        assert restored.choose(RULES, set()) == persistent.choose(RULES, set())
        persistent = restored
        results.append(row)

        if episode % 10 == 0:
            cold = sum(r["cold"]["questions"] for r in results)
            warm = sum(r["persistent"]["questions"] for r in results)
            print(
                f"Episode {episode}/60 | questions cumulees : "
                f"sans_historique={cold}, avec_bibliotheque={warm}",
                flush=True,
            )

    summary = {}
    for label in ("cold", "persistent"):
        summary[label] = {
            "total_questions": sum(r[label]["questions"] for r in results),
            "correct_unqueried": sum(r[label]["correct_unqueried"] for r in results),
            "total_unqueried": sum(r[label]["total_unqueried"] for r in results),
        }
        print(f"{label} : {summary[label]}")

    saved = (
        summary["cold"]["total_questions"]
        - summary["persistent"]["total_questions"]
    )
    print(f"Questions economisees : {saved}")
    print(f"Regles consolidees : {len(persistent.counts)}")
    print(f"Grammaire : {len(RULES)} regles possibles")

    (output / "results.json").write_text(
        json.dumps({
            "config": {
                "episodes": 60,
                "seed": 71001,
                "frequent_rule_probability": 0.8,
                "frequent_rule_count": 3,
                "grammar_size": len(RULES),
                "stop_condition": "unique_compatible_rule",
                "library_weight": "one_plus_previous_identifications",
            },
            "summary": summary,
            "questions_saved": saved,
            "episodes": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
