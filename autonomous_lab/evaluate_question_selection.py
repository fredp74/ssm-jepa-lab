import itertools
import json
from pathlib import Path

import torch

from meta_ssm import MetaSSM, INPUTS, environment
from context_holdout import context_rank

torch.set_num_threads(1)

@torch.no_grad()
def main():
    root = Path("autonomous_lab/runs/context_holdout_ssm_01")
    output = Path("autonomous_lab/runs/frozen_question_selection_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    protocol = json.loads(
        (root / "protocol.json").read_text(encoding="utf-8")
    )
    contexts = protocol["test_contexts"]
    informative = [c for c in contexts if context_rank(c) == 4]
    if len(contexts) != 20 or len(informative) != 16:
        raise SystemExit("Partage de contextes inattendu.")

    output.mkdir(parents=True)
    inputs = torch.tensor(INPUTS)
    permutations = torch.tensor(
        list(itertools.permutations(range(4)))
    )
    results = []

    for seed in protocol["seeds"]:
        model = MetaSSM()
        model.load_state_dict(torch.load(
            root / f"model_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        ))
        model.eval()

        context_results = []
        for context in contexts:
            # 16 regles x 24 ordres d'observation.
            ordered_x = inputs[torch.tensor(context)][permutations]
            observed_x = ordered_x.repeat(16, 1, 1)
            rules = torch.arange(16).repeat_interleave(24)
            observed_y = environment(rules, observed_x)

            remaining = [i for i in range(8) if i not in context]
            queries = inputs[remaining][None].expand(
                len(rules), -1, -1
            )
            expected = environment(rules, queries)
            predictions = model(
                observed_x, observed_y, queries
            ).argmax(-1)

            context_results.append({
                "context": context,
                "rank": context_rank(context),
                "correct": int((predictions == expected).sum()),
                "total": expected.numel(),
            })

        summaries = {}
        for policy in ("random", "informative"):
            selected = [
                row for row in context_results
                if policy == "random" or row["rank"] == 4
            ]
            correct = sum(row["correct"] for row in selected)
            total = sum(row["total"] for row in selected)
            symbolic_score = sum(
                row["total"] * (1.0 if row["rank"] == 4 else 0.5)
                for row in selected
            )
            summaries[policy] = {
                "contexts": len(selected),
                "correct": correct,
                "total": total,
                "symbolic_expected_score": symbolic_score,
            }
            print(
                f"Graine {seed} | {policy} | "
                f"SSM={correct}/{total} ({100*correct/total:.2f} %) | "
                f"reference_symbolique={100*symbolic_score/total:.1f} %",
                flush=True,
            )

        results.append({
            "seed": seed,
            "summaries": summaries,
            "context_results": context_results,
        })

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "source_protocol": str(root / "protocol.json"),
                "questions_per_episode": 4,
                "evaluation_updates": 0,
                "orders_per_context": 24,
                "random_policy": "uniform_over_20_reserved_contexts",
                "informative_policy": "uniform_over_16_rank4_reserved_contexts",
                "policy_knowledge": "known_affine_rule_family",
                "torch_version": str(torch.__version__),
            },
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
