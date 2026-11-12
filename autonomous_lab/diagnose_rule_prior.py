import json
from pathlib import Path

import torch

INPUTS = [
    (a, b, c)
    for a in (0, 1)
    for b in (0, 1)
    for c in (0, 1)
]
TEST_RULES = [3, 5, 10, 12]
TRAIN_RULES = [r for r in range(16) if r not in TEST_RULES]

def answer(rule, x):
    coefficients = [(rule >> i) & 1 for i in range(4)]
    return (
        coefficients[0]
        + sum(a*b for a, b in zip(coefficients[1:], x))
    ) % 2

def evaluate(actual_rules, prior, count):
    generator = torch.Generator().manual_seed(51999)
    stats = {
        "episodes": 0,
        "incompatible_episodes": 0,
        "questions": 0,
        "answered": 0,
        "abstentions": 0,
        "ties": 0,
        "score": 0.0,
    }

    for actual in actual_rules:
        for _ in range(24):
            order = torch.randperm(8, generator=generator).tolist()
            observed = order[:count]
            remaining = order[count:]

            compatible = [
                candidate for candidate in prior
                if all(
                    answer(candidate, INPUTS[i]) == answer(actual, INPUTS[i])
                    for i in observed
                )
            ]

            stats["episodes"] += 1
            stats["questions"] += len(remaining)

            if not compatible:
                stats["incompatible_episodes"] += 1
                stats["abstentions"] += len(remaining)
                continue

            for i in remaining:
                stats["answered"] += 1
                votes = sum(
                    answer(candidate, INPUTS[i])
                    for candidate in compatible
                )
                if 2*votes == len(compatible):
                    stats["ties"] += 1
                    stats["score"] += 0.5
                else:
                    prediction = int(2*votes > len(compatible))
                    stats["score"] += (
                        prediction == answer(actual, INPUTS[i])
                    )

    return stats

def main():
    output = Path("autonomous_lab/runs/prior_diagnostic_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    results = []
    for split, actual_rules in (
        ("train", TRAIN_RULES),
        ("test", TEST_RULES),
    ):
        for label, prior in (
            ("12_regles", TRAIN_RULES),
            ("16_regles", list(range(16))),
        ):
            print(f"\n{split.upper()} | {label}")
            for count in (0, 1, 2, 4, 8):
                stats = evaluate(actual_rules, prior, count)
                results.append({
                    "split": split,
                    "prior": label,
                    "observations": count,
                    **stats,
                })

                accuracy = (
                    f"{100*stats['score']/stats['answered']:.1f} %"
                    if stats["answered"] else "non applicable"
                )
                print(
                    f"observations={count} | "
                    f"score_sur_reponses={accuracy} | "
                    f"couverture={stats['answered']}/{stats['questions']} | "
                    f"egalites={stats['ties']} | "
                    f"episodes_incompatibles="
                    f"{stats['incompatible_episodes']}/{stats['episodes']}"
                )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "train_rules": TRAIN_RULES,
            "test_rules": TEST_RULES,
            "tie_score": 0.5,
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
