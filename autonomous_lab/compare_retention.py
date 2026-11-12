import json
from pathlib import Path

import torch
from torch.nn import functional as F

from meta_ssm import MetaSSM, INPUTS, QUESTIONS, environment

torch.set_num_threads(1)

TEST_RULES = [3, 5, 10, 12]
TRAIN_RULES = [r for r in range(16) if r not in TEST_RULES]
SEEDS = (51001, 51002, 51003)

@torch.no_grad()
def evaluate(model, rules):
    model.eval()
    results = {}
    inputs = torch.tensor(INPUTS)

    for count in (0, 1, 2, 4, 8):
        stats = {
            "seen_correct": 0, "seen_total": 0,
            "unseen_correct": 0, "unseen_total": 0,
        }
        generator = torch.Generator().manual_seed(51999)

        for rule in rules:
            rule_ids = torch.tensor([rule])
            for _ in range(24):
                order = torch.randperm(8, generator=generator)
                x = inputs[order[:count]][None]
                y = environment(rule_ids, x)
                queries = inputs[None]
                expected = environment(rule_ids, queries)
                answer = model(x, y, queries).argmax(-1)[0]
                correct = answer == expected[0]

                observed = order[:count]
                remaining = order[count:]
                stats["seen_correct"] += int(correct[observed].sum())
                stats["seen_total"] += count
                stats["unseen_correct"] += int(correct[remaining].sum())
                stats["unseen_total"] += 8-count

        results[str(count)] = stats
    return results


@torch.no_grad()
def recency(model, rules):
    model.eval()
    inputs = torch.tensor(INPUTS)
    generator = torch.Generator().manual_seed(52999)
    correct = [0] * 8
    totals = [0] * 8

    for rule in rules:
        ids = torch.tensor([rule])
        for _ in range(24):
            order = torch.randperm(8, generator=generator)
            x = inputs[order][None]
            y = environment(ids, x)
            answers = model(x, y, x).argmax(-1)[0]
            for position in range(8):
                correct[position] += int(
                    answers[position] == y[0, position]
                )
                totals[position] += 1

    return {
        "correct_by_position": correct,
        "total_by_position": totals,
    }


def main():
    output = Path("autonomous_lab/runs/retention_comparison_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    inputs = torch.tensor(INPUTS)
    train_ids = torch.tensor(TRAIN_RULES)
    results = []

    for seed in SEEDS:
        for arm in ("standard", "retention"):
            torch.manual_seed(seed)
            model = MetaSSM()
            if arm == "retention":
                with torch.no_grad():
                    model.decay.fill_(2.9444389791664403)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Generateurs independants : memes regles par pas
            # dans les deux branches.
            rule_rng = torch.Generator().manual_seed(seed + 100)
            question_rng = torch.Generator().manual_seed(seed + 200)

            for step in range(1500):
                ids = train_ids[torch.randint(
                    len(train_ids), (32,), generator=rule_rng
                )]

                if False:  # Les deux branches utilisent le protocole variable.
                    order = torch.rand(
                        32, 4, generator=question_rng
                    ).argsort(dim=1)
                    x = torch.tensor(QUESTIONS)[order]
                else:
                    count = step % 9
                    order = torch.rand(
                        32, 8, generator=question_rng
                    ).argsort(dim=1)
                    x = inputs[order[:, :count]]

                y = environment(ids, x)
                queries = inputs[None].expand(32, -1, -1)
                expected = environment(ids, queries)

                model.train()
                logits = model(x, y, queries)
                loss = F.cross_entropy(
                    logits.reshape(-1, 2), expected.reshape(-1)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Perte non finie.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()

                if (step + 1) % 500 == 0:
                    print(
                        f"{arm} | graine {seed} | {step+1}/1500 | "
                        f"perte={loss.item():.4f}",
                        flush=True,
                    )

            result = {
                "seed": seed,
                "arm": arm,
                "train": evaluate(model, TRAIN_RULES),
                "test": evaluate(model, TEST_RULES),
                "train_recency": recency(model, TRAIN_RULES),
                "test_recency": recency(model, TEST_RULES),
                "final_decay": model.decay.detach().sigmoid().tolist(),
            }
            results.append(result)
            torch.save(
                model.state_dict(), output / f"{arm}_{seed}.pt"
            )

            print(f"TEST | {arm} | graine {seed}", flush=True)
            for count, stats in result["test"].items():
                print(
                    f"  observations={count} | "
                    f"non_observees={stats['unseen_correct']}/"
                    f"{stats['unseen_total']} | "
                    f"observees={stats['seen_correct']}/"
                    f"{stats['seen_total']}",
                    flush=True,
                )

            # Sauvegarde progressive des resultats.
            (output / "results.json").write_text(
                json.dumps({
                    "config": {
                        "initial_decay": {"standard": 0.5, "retention": 0.95},
                        "decay_trainable": True,
                        "train_rules": TRAIN_RULES,
                        "test_rules": TEST_RULES,
                        "seeds": list(SEEDS),
                        "steps": 1500,
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "evaluation_updates": 0,
                        "evaluation_draws_per_rule": 24,
                        "torch_version": str(torch.__version__),
                    },
                    "results": results,
                }, indent=2),
                encoding="utf-8",
            )

    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
