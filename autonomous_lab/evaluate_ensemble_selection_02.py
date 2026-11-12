import json
import random
from pathlib import Path

import torch
from meta_ssm import MetaSSM, INPUTS, environment
from context_holdout import context_rank

torch.set_num_threads(1)

def entropy(probabilities):
    return -(
        probabilities * probabilities.clamp_min(1e-12).log2()
    ).sum(-1)

@torch.no_grad()
def predict(models, questions, answers, queries):
    x = torch.tensor(questions, dtype=torch.long).reshape(1, -1, 3)
    y = torch.tensor(answers, dtype=torch.long).reshape(1, -1)
    q = torch.tensor(queries, dtype=torch.long).reshape(1, -1, 3)
    individual = torch.stack([
        model(x, y, q).softmax(-1)[0] for model in models
    ])
    return individual, individual.mean(0)

@torch.no_grad()
def main():
    root = Path("autonomous_lab/runs/partial_context_ensemble_02")
    output = Path("autonomous_lab/runs/ensemble_selection_02")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    protocol = json.loads(
        (root / "protocol.json").read_text(encoding="utf-8")
    )
    train_contexts = {
        tuple(sorted(c)) for c in protocol["train_contexts"]
    }

    models = []
    for seed in protocol["ensemble_seeds"]:
        model = MetaSSM()
        model.load_state_dict(torch.load(
            root / f"model_{seed}.pt",
            map_location="cpu", weights_only=True,
        ))
        model.eval()
        models.append(model)

    output.mkdir(parents=True)
    summaries, episodes = [], []

    for policy in ("random", "entropy", "disagreement"):
        stats = {
            "episodes": 0,
            "correct": 0,
            "total": 0,
            "informative_contexts": 0,
            "training_contexts": 0,
        }

        for rule in range(16):
            for repetition in range(24):
                rng = random.Random(63001000 + repetition)
                questions, answers, choices = [], [], []

                for step in range(4):
                    available = [x for x in INPUTS if x not in questions]

                    if policy == "random":
                        question = rng.choice(available)
                    else:
                        individual, mean = predict(
                            models, questions, answers, available
                        )
                        scores = entropy(mean)
                        if policy == "disagreement":
                            scores = scores - entropy(individual).mean(0)

                        maximum = float(scores.max())
                        candidates = [
                            i for i, value in enumerate(scores.tolist())
                            if maximum-value <= 1e-7
                        ]
                        selected = rng.choice(candidates)
                        question = available[selected]
                        choices.append({
                            "step": step,
                            "selected_score": float(scores[selected]),
                        })

                    answer = int(environment(
                        torch.tensor([rule]),
                        torch.tensor([[question]]),
                    )[0, 0])
                    questions.append(question)
                    answers.append(answer)

                remaining = [x for x in INPUTS if x not in questions]
                _, mean = predict(models, questions, answers, remaining)
                predictions = mean.argmax(-1).tolist()
                expected = environment(
                    torch.tensor([rule]),
                    torch.tensor([remaining]),
                )[0].tolist()
                correct = sum(
                    a == b for a, b in zip(predictions, expected)
                )
                context = tuple(sorted(
                    INPUTS.index(x) for x in questions
                ))

                stats["episodes"] += 1
                stats["correct"] += correct
                stats["total"] += 4
                stats["informative_contexts"] += context_rank(context) == 4
                stats["training_contexts"] += context in train_contexts

                episodes.append({
                    "policy": policy,
                    "rule": rule,
                    "repetition": repetition,
                    "questions": questions,
                    "answers": answers,
                    "remaining": remaining,
                    "expected": expected,
                    "predictions": predictions,
                    "correct": correct,
                    "context_rank": context_rank(context),
                    "context_seen_in_training": context in train_contexts,
                    "selection_scores": choices,
                })

        summaries.append({"policy": policy, **stats})
        print(
            f"{policy} | score={stats['correct']}/{stats['total']} "
            f"({100*stats['correct']/stats['total']:.2f} %) | "
            f"contextes_informatifs={stats['informative_contexts']}/"
            f"{stats['episodes']} | "
            f"contextes_train={stats['training_contexts']}/"
            f"{stats['episodes']}",
            flush=True,
        )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "ensemble_seeds": protocol["ensemble_seeds"],
                "ensemble_count": 1,
                "budget": 4,
                "repetitions_per_rule": 24,
                "evaluation_updates": 0,
                "final_prediction": "mean_member_probabilities",
                "disagreement": "entropy_of_mean_minus_mean_entropy",
            },
            "summaries": summaries,
            "episodes": episodes,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
