import json
import random
from pathlib import Path

import torch

from baseline import Agent
from meta_ssm import MetaSSM, INPUTS, environment
from context_holdout import context_rank

torch.set_num_threads(1)

@torch.no_grad()
def model_probabilities(model, questions, answers, queries):
    x = torch.tensor(questions, dtype=torch.long).reshape(1, -1, 3)
    y = torch.tensor(answers, dtype=torch.long).reshape(1, -1)
    q = torch.tensor(queries, dtype=torch.long).reshape(1, -1, 3)
    return model(x, y, q).softmax(-1)[0]

@torch.no_grad()
def main():
    root = Path("autonomous_lab/runs/context_holdout_ssm_01")
    output = Path("autonomous_lab/runs/sequential_selection_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    protocol = json.loads(
        (root / "protocol.json").read_text(encoding="utf-8")
    )
    train_contexts = {
        tuple(sorted(context)) for context in protocol["train_contexts"]
    }
    test_contexts = {
        tuple(sorted(context)) for context in protocol["test_contexts"]
    }
    output.mkdir(parents=True)

    summaries = []
    episodes = []

    for seed in protocol["seeds"]:
        model = MetaSSM()
        model.load_state_dict(torch.load(
            root / f"model_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        ))
        model.eval()

        for policy in ("random", "uncertainty", "symbolic"):
            stats = {
                "episodes": 0,
                "correct": 0,
                "total": 0,
                "informative_contexts": 0,
                "training_contexts": 0,
                "heldout_contexts": 0,
            }

            for rule in range(16):
                for repetition in range(24):
                    # L'environnement cache son identifiant au modele.
                    rng = random.Random(seed*1000 + repetition)
                    questions, answers = [], []
                    symbolic = Agent()

                    for step in range(4):
                        available = [x for x in INPUTS if x not in questions]

                        if policy == "random":
                            question = rng.choice(available)
                        elif policy == "symbolic":
                            question = symbolic.choose_question()
                        else:
                            probabilities = model_probabilities(
                                model, questions, answers, available
                            )
                            entropy = -(
                                probabilities
                                * probabilities.clamp_min(1e-12).log2()
                            ).sum(-1)
                            maximum = float(entropy.max())
                            candidates = [
                                i for i, value in enumerate(entropy.tolist())
                                if maximum-value <= 1e-7
                            ]
                            question = available[rng.choice(candidates)]

                        response = int(environment(
                            torch.tensor([rule]),
                            torch.tensor([[question]]),
                        )[0, 0])

                        questions.append(question)
                        answers.append(response)
                        symbolic.observe(question, response)

                    remaining = [x for x in INPUTS if x not in questions]
                    probabilities = model_probabilities(
                        model, questions, answers, remaining
                    )
                    predictions = probabilities.argmax(-1).tolist()
                    expected = environment(
                        torch.tensor([rule]),
                        torch.tensor([remaining]),
                    )[0].tolist()
                    correct = sum(
                        a == b for a, b in zip(predictions, expected)
                    )

                    context = tuple(sorted(
                        INPUTS.index(question) for question in questions
                    ))
                    informative = context_rank(context) == 4
                    stats["episodes"] += 1
                    stats["correct"] += correct
                    stats["total"] += len(remaining)
                    stats["informative_contexts"] += informative
                    stats["training_contexts"] += context in train_contexts
                    stats["heldout_contexts"] += context in test_contexts

                    episodes.append({
                        "seed": seed,
                        "policy": policy,
                        "rule": rule,
                        "repetition": repetition,
                        "questions": questions,
                        "answers": answers,
                        "context_rank": context_rank(context),
                        "context_seen_in_training": context in train_contexts,
                        "remaining": remaining,
                        "expected": expected,
                        "predictions": predictions,
                        "correct": correct,
                    })

            summaries.append({"seed": seed, "policy": policy, **stats})
            print(
                f"Graine {seed} | {policy} | "
                f"score={stats['correct']}/{stats['total']} "
                f"({100*stats['correct']/stats['total']:.2f} %) | "
                f"contextes_informatifs={stats['informative_contexts']}/"
                f"{stats['episodes']} | "
                f"contextes_reserves={stats['heldout_contexts']}/"
                f"{stats['episodes']}",
                flush=True,
            )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "source_protocol": str(root / "protocol.json"),
                "budget": 4,
                "repetitions_per_rule": 24,
                "evaluation_updates": 0,
                "uncertainty": "maximum_binary_predictive_entropy",
                "symbolic_policy": "known_family_information_gain",
                "partial_contexts_absent_from_training": True,
                "evaluation_includes_training_and_heldout_contexts": True,
            },
            "summaries": summaries,
            "episodes": episodes,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
