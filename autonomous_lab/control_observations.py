import itertools
import json
from pathlib import Path

import torch
from meta_ssm import MetaSSM, QUESTIONS, UNASKED, TEST_RULES, environment

torch.set_num_threads(1)

@torch.no_grad()
def main():
    root = Path("autonomous_lab/runs/meta_ssm_01")
    output = root / "observation_controls.json"
    if output.exists():
        raise SystemExit("Resultats deja presents : aucun remplacement.")

    summaries = []
    details = []

    for seed in (41001, 41002, 41003):
        model = MetaSSM()
        model.load_state_dict(torch.load(
            root / f"model_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        ))
        model.eval()

        counts = {
            "baseline_correct": 0,
            "baseline_total": 0,
            "changed_predictions": 0,
            "follows_replacement": 0,
            "matches_original": 0,
            "original_replacement_agreement": 0,
            "total": 0,
        }
        probability_shift = 0.0

        for rule in TEST_RULES:
            for order in itertools.permutations(range(4)):
                x = torch.tensor([[
                    QUESTIONS[i] for i in order
                ]])
                queries = torch.tensor([UNASKED])
                original_id = torch.tensor([rule])
                y = environment(original_id, x)
                expected_original = environment(original_id, queries)

                original_prob = model(x, y, queries).softmax(-1)
                original_answer = original_prob.argmax(-1)
                counts["baseline_correct"] += int(
                    (original_answer == expected_original).sum()
                )
                counts["baseline_total"] += len(UNASKED)

                for replacement in TEST_RULES:
                    if replacement == rule:
                        continue

                    replacement_id = torch.tensor([replacement])
                    replacement_y = environment(replacement_id, x)
                    expected_replacement = environment(
                        replacement_id, queries
                    )
                    probabilities = model(
                        x, replacement_y, queries
                    ).softmax(-1)
                    answer = probabilities.argmax(-1)

                    changed = int((answer != original_answer).sum())
                    follows = int(
                        (answer == expected_replacement).sum()
                    )
                    matches_old = int(
                        (answer == expected_original).sum()
                    )
                    shift = float(
                        (probabilities-original_prob).abs().sum(-1).sum()/2
                    )

                    counts["changed_predictions"] += changed
                    counts["follows_replacement"] += follows
                    counts["matches_original"] += matches_old
                    counts["original_replacement_agreement"] += int(
                        (expected_original == expected_replacement).sum()
                    )
                    counts["total"] += len(UNASKED)
                    probability_shift += shift

                    details.append({
                        "seed": seed,
                        "original_rule": rule,
                        "replacement_rule": replacement,
                        "order": list(order),
                        "original_predictions": original_answer[0].tolist(),
                        "replacement_predictions": answer[0].tolist(),
                        "expected_original": expected_original[0].tolist(),
                        "expected_replacement": expected_replacement[0].tolist(),
                    })

        summary = {
            "seed": seed,
            **counts,
            "mean_total_variation": probability_shift/counts["total"],
        }
        summaries.append(summary)

        n = counts["total"]
        print(
            f"Graine {seed} | "
            f"reference={counts['baseline_correct']}/"
            f"{counts['baseline_total']} | "
            f"predictions_changees={counts['changed_predictions']}/{n} | "
            f"suit_remplacement={counts['follows_replacement']}/{n} | "
            f"correspond_ancienne={counts['matches_original']}/{n} | "
            f"distance_moyenne={summary['mean_total_variation']:.4f}",
            flush=True,
        )

    output.write_text(
        json.dumps({"summaries": summaries, "details": details}, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {output}")

if __name__ == "__main__":
    main()
