import argparse
import itertools
import json
import random
from pathlib import Path

INPUTS = list(itertools.product((0, 1), repeat=3))
RULES = list(itertools.product((0, 1), repeat=4))

def predict(rule, x):
    # Famille connue : biais XOR combinaison des trois bits.
    return (rule[0] + sum(a*b for a, b in zip(rule[1:], x))) % 2

class Environment:
    def __init__(self, seed):
        self._rule = random.Random(seed).choice(RULES)

    def answer(self, x):
        return predict(self._rule, x)

class Agent:
    def __init__(self):
        self.observations = []

    def hypotheses(self):
        return [
            rule for rule in RULES
            if all(
                predict(rule, tuple(item["input"])) == item["output"]
                for item in self.observations
            )
        ]

    def choose_question(self):
        hypotheses = self.hypotheses()
        asked = {tuple(item["input"]) for item in self.observations}
        available = [x for x in INPUTS if x not in asked]

        # Choisir une question qui partage les hypotheses aussi
        # equitablement que possible entre les deux reponses.
        def score(x):
            ones = sum(predict(rule, x) for rule in hypotheses)
            return min(ones, len(hypotheses)-ones)

        return max(available, key=score)

    def observe(self, x, answer):
        self.observations.append({"input": list(x), "output": answer})
        if not self.hypotheses():
            raise ValueError("Observations incompatibles avec la famille connue.")

    def answer(self, x):
        votes = [predict(rule, x) for rule in self.hypotheses()]
        return int(sum(votes) > len(votes)/2)

    def save(self, path):
        path.write_text(
            json.dumps({
                "format": 1,
                "family": "affine_binary_3",
                "observations": self.observations,
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["format"] != 1 or data["family"] != "affine_binary_3":
            raise ValueError("Memoire incompatible.")
        agent = cls()
        for item in data["observations"]:
            agent.observe(tuple(item["input"]), item["output"])
        return agent

def evaluate(agent, environment, inputs):
    return sum(agent.answer(x) == environment.answer(x) for x in inputs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("autonomous_lab/runs/baseline_01"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes doit etre positif")
    if args.out.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    args.out.mkdir(parents=True)

    results = []
    for episode in range(args.episodes):
        environment = Environment(args.seed + episode)
        agent = Agent()

        for _ in range(4):
            question = agent.choose_question()
            agent.observe(question, environment.answer(question))

        asked = {tuple(item["input"]) for item in agent.observations}
        heldout = [x for x in INPUTS if x not in asked]

        # Comparaison avant/apres sur les memes entrees reservees.
        before = evaluate(Agent(), environment, heldout)
        after = evaluate(agent, environment, heldout)

        path = args.out / f"memory_{episode:03d}.json"
        agent.save(path)
        restored = Agent.load(path)
        reloaded = evaluate(restored, environment, heldout)
        assert all(agent.answer(x) == restored.answer(x) for x in INPUTS)

        result = {
            "episode": episode,
            "questions": len(asked),
            "remaining_hypotheses": len(agent.hypotheses()),
            "heldout_count": len(heldout),
            "correct_before": before,
            "correct_after": after,
            "correct_reloaded": reloaded,
        }
        results.append(result)
        print(
            f"Episode {episode+1} | questions={len(asked)} | "
            f"hypotheses={len(agent.hypotheses())} | "
            f"avant={before}/4 | apres={after}/4 | recharge={reloaded}/4"
        )

    (args.out / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    total = sum(row["heldout_count"] for row in results)
    for key in ("correct_before", "correct_after", "correct_reloaded"):
        print(f"{key}: {sum(row[key] for row in results)}/{total}")
    print(f"Resultats et memoires : {args.out}")

if __name__ == "__main__":
    main()
