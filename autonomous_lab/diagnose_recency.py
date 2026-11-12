import json
from pathlib import Path

import torch
from meta_ssm import MetaSSM, INPUTS, environment

torch.set_num_threads(1)

TRAIN_RULES = [0, 1, 2, 4, 6, 7, 8, 9, 11, 13, 14, 15]
TEST_RULES = [3, 5, 10, 12]

@torch.no_grad()
def main():
    root = Path("autonomous_lab/runs/observation_curriculum_01")
    output = root / "recency_diagnostic.json"
    if output.exists():
        raise SystemExit("Diagnostic deja present : aucun remplacement.")

    inputs = torch.tensor(INPUTS)
    results = []

    for seed in (51001, 51002, 51003):
        model = MetaSSM()
        model.load_state_dict(torch.load(
            root / f"variable_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        ))
        model.eval()

        for split, rules in (("train", TRAIN_RULES), ("test", TEST_RULES)):
            generator = torch.Generator().manual_seed(51999)
            correct = [0] * 8
            totals = [0] * 8

            for rule in rules:
                rule_ids = torch.tensor([rule])
                for _ in range(24):
                    order = torch.randperm(8, generator=generator)
                    observed_x = inputs[order][None]
                    observed_y = environment(rule_ids, observed_x)

                    # Interroger toutes les entrees apres avoir lu
                    # les huit observations. Aucun nouvel apport au contexte.
                    queries = inputs[None]
                    predictions = model(
                        observed_x, observed_y, queries
                    ).argmax(-1)[0]

                    for position, input_id in enumerate(order.tolist()):
                        expected = int(observed_y[0, position])
                        correct[position] += (
                            int(predictions[input_id]) == expected
                        )
                        totals[position] += 1

            results.append({
                "seed": seed,
                "split": split,
                "correct_by_position": correct,
                "total_by_position": totals,
            })

    summaries = {}
    for split in ("train", "test"):
        rows = [row for row in results if row["split"] == split]
        correct = [
            sum(row["correct_by_position"][i] for row in rows)
            for i in range(8)
        ]
        totals = [
            sum(row["total_by_position"][i] for row in rows)
            for i in range(8)
        ]
        summaries[split] = {
            "correct_by_position": correct,
            "total_by_position": totals,
        }

        print(f"\n{split.upper()} — moyenne des trois graines")
        for position, (good, total) in enumerate(zip(correct, totals), 1):
            print(
                f"Observation {position} | {good}/{total} | "
                f"{100*good/total:.1f} %"
            )

    output.write_text(
        json.dumps({
            "results": results,
            "summaries": summaries,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {output}")

if __name__ == "__main__":
    main()
