import itertools
import json
from pathlib import Path

import torch
from meta_ssm import MetaSSM, INPUTS, environment
from context_holdout import context_rank

torch.set_num_threads(1)

@torch.no_grad()
def main():
    root = Path("autonomous_lab/runs/partial_context_ensemble_03")
    output = Path("autonomous_lab/runs/exhaustive_ensemble_random_03")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    protocol = json.loads(
        (root / "protocol.json").read_text(encoding="utf-8")
    )
    models = []
    for seed in protocol["ensemble_seeds"]:
        model = MetaSSM()
        model.load_state_dict(torch.load(
            root / f"model_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        ))
        model.eval()
        models.append(model)

    inputs = torch.tensor(INPUTS)
    orders = torch.tensor(list(itertools.permutations(range(4))))
    rules = torch.arange(16).repeat_interleave(24)
    results = []

    for number, context in enumerate(
        itertools.combinations(range(8), 4), 1
    ):
        x = inputs[torch.tensor(context)][orders].repeat(16, 1, 1)
        y = environment(rules, x)
        remaining = [i for i in range(8) if i not in context]
        queries = inputs[remaining][None].expand(len(rules), -1, -1)
        expected = environment(rules, queries)

        probabilities = torch.stack([
            model(x, y, queries).softmax(-1)
            for model in models
        ]).mean(0)
        predictions = probabilities.argmax(-1)
        correct = (predictions == expected).reshape(16, 24, 4)

        results.append({
            "context": list(context),
            "rank": context_rank(context),
            "correct": int(correct.sum()),
            "total": correct.numel(),
            "correct_by_rule": correct.sum(dim=(1, 2)).tolist(),
        })

        if number % 10 == 0:
            print(f"Contextes evalues : {number}/70", flush=True)

    total_correct = sum(row["correct"] for row in results)
    total = sum(row["total"] for row in results)
    baseline = total_correct / total
    summary = {
        "correct": total_correct,
        "total": total,
        "accuracy": baseline,
        "contexts": len(results),
        "informative_contexts": sum(row["rank"] == 4 for row in results),
    }

    print(
        f"\nHASARD EXHAUSTIF : {total_correct}/{total} "
        f"({100*baseline:.3f} %)"
    )
    for rank in (3, 4):
        rows = [row for row in results if row["rank"] == rank]
        good = sum(row["correct"] for row in rows)
        count = sum(row["total"] for row in rows)
        print(f"Rang {rank} : {good}/{count} ({100*good/count:.3f} %)")

    previous = Path(
        "autonomous_lab/runs/ensemble_selection_03/results.json"
    )
    comparisons = []
    if previous.exists():
        data = json.loads(previous.read_text(encoding="utf-8"))
        for row in data["summaries"]:
            accuracy = row["correct"] / row["total"]
            comparison = {
                "policy": row["policy"],
                "accuracy": accuracy,
                "difference_percentage_points": 100*(accuracy-baseline),
            }
            comparisons.append(comparison)
            print(
                f"{row['policy']} : {100*accuracy:.3f} % | "
                f"ecart au hasard exhaustif : "
                f"{100*(accuracy-baseline):+.3f} points"
            )

    output.mkdir(parents=True)
    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "ensemble_seeds": protocol["ensemble_seeds"],
                "rules": 16,
                "contexts": 70,
                "orders_per_context": 24,
                "evaluation_updates": 0,
                "random_policy": "uniform_ordered_sampling_without_replacement",
            },
            "summary": summary,
            "comparisons": comparisons,
            "context_results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
