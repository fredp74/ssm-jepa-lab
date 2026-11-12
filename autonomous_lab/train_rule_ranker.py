import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)

class RuleRanker(nn.Module):
    def __init__(self, classes, width=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, width), nn.Tanh())
        self.decay = nn.Parameter(torch.zeros(width))
        self.head = nn.Linear(width, classes)

    def forward(self, x, lengths):
        h = torch.zeros(x.size(0), self.decay.numel(), device=x.device)
        decay = self.decay.sigmoid()
        for position in range(x.size(1)):
            updated = decay*h + self.encoder(x[:, position])
            h = torch.where(
                (position < lengths)[:, None], updated, h
            )
        return self.head(h)

def load_data(path, classes):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    x = torch.zeros(len(rows), 12, 6)
    lengths = torch.zeros(len(rows), dtype=torch.long)
    targets = torch.zeros(len(rows), classes)

    for i, row in enumerate(rows):
        lengths[i] = len(row["observations"])
        for j, item in enumerate(row["observations"]):
            x[i, j] = torch.tensor(
                item["input"] + [item["output"]], dtype=torch.float
            )*2-1
        compatible = row["compatible_rule_ids"]
        targets[i, compatible] = 1/len(compatible)

    return rows, x, lengths, targets

@torch.no_grad()
def evaluate(model, rows, x, lengths, rules):
    model.eval()
    orders = []
    for start in range(0, len(rows), 256):
        logits = model(x[start:start+256], lengths[start:start+256])
        orders.extend(
            logits.argsort(dim=-1, descending=True, stable=True).tolist()
        )

    def check_candidate(rule_id, observations):
        return all(
            rules[rule_id][item["input_index"]] == item["output"]
            for item in observations
        )

    details = []
    for row, neural_order in zip(rows, orders):
        result = {
            "observations": len(row["observations"]),
            "compatible_rules": len(row["compatible_rule_ids"]),
        }
        observed = {item["input_index"] for item in row["observations"]}
        truth = rules[row["generating_rule_id"]]
        remaining = [i for i in range(32) if i not in observed]

        for label, order in (
            ("fixed", list(range(len(rules)))),
            ("ssm", neural_order),
        ):
            checked = 0
            for candidate in order:
                checked += 1
                if check_candidate(candidate, row["observations"]):
                    break
            else:
                raise RuntimeError("Aucune candidate compatible.")

            result[label] = {
                "candidates_checked": checked,
                "true_rule_rank": order.index(row["generating_rule_id"])+1,
                "selected_true_rule": candidate == row["generating_rule_id"],
                "correct_unqueried": sum(
                    rules[candidate][i] == truth[i] for i in remaining
                ),
                "total_unqueried": len(remaining),
            }
        details.append(result)

    summaries = {}
    for count in (4, 8, 12):
        selected = [row for row in details if row["observations"] == count]
        summaries[str(count)] = {}
        for label in ("fixed", "ssm"):
            summaries[str(count)][label] = {
                "episodes": len(selected),
                "mean_candidates_checked": sum(
                    row[label]["candidates_checked"] for row in selected
                ) / len(selected),
                "mean_true_rule_rank": sum(
                    row[label]["true_rule_rank"] for row in selected
                ) / len(selected),
                "selected_true_rule": sum(
                    row[label]["selected_true_rule"] for row in selected
                ),
                "correct_unqueried": sum(
                    row[label]["correct_unqueried"] for row in selected
                ),
                "total_unqueried": sum(
                    row[label]["total_unqueried"] for row in selected
                ),
            }
    return {"summaries": summaries, "details": details}

def main():
    data_root = Path("autonomous_lab/data/rule_ranking_01")
    output = Path("autonomous_lab/runs/rule_ranking_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    protocol = json.loads(
        (data_root / "protocol.json").read_text(encoding="utf-8")
    )
    rules = [tuple(map(int, rule)) for rule in protocol["rules"]]
    classes = len(rules)
    _, train_x, train_lengths, targets = load_data(
        data_root / "train.jsonl", classes
    )
    rows, validation_x, validation_lengths, _ = load_data(
        data_root / "validation.jsonl", classes
    )

    config = {
        "seeds": [91011, 91012, 91013],
        "steps": 1500,
        "batch_size": 64,
        "learning_rate": 0.001,
        "target": "uniform_over_symbolically_compatible_rules",
        "verifier": "exact_observation_consistency",
        "baseline": "fixed_rule_order",
        "timing_benchmark_included": False,
        "data_sha256": {
            name: hashlib.sha256(
                (data_root / name).read_bytes()
            ).hexdigest()
            for name in ("train.jsonl", "validation.jsonl", "protocol.json")
        },
        "torch_version": str(torch.__version__),
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    results = []
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed+100)
        model = RuleRanker(classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for step in range(1500):
            indices = torch.randint(len(train_x), (64,), generator=rng)
            model.train()
            logits = model(train_x[indices], train_lengths[indices])
            loss = -(
                targets[indices] * F.log_softmax(logits, dim=-1)
            ).sum(-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Perte non finie.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()

            if (step+1) % 500 == 0:
                print(
                    f"Graine {seed} | {step+1}/1500 | perte={loss.item():.4f}",
                    flush=True,
                )

        evaluation = evaluate(
            model, rows, validation_x, validation_lengths, rules
        )
        results.append({"seed": seed, **evaluation})
        torch.save(model.state_dict(), output / f"model_{seed}.pt")

        for count, scores in evaluation["summaries"].items():
            print(
                f"Graine {seed} | observations={count} | "
                f"candidates testees : "
                f"fixe={scores['fixed']['mean_candidates_checked']:.2f}, "
                f"SSM={scores['ssm']['mean_candidates_checked']:.2f}",
                flush=True,
            )

        (output / "results.json").write_text(
            json.dumps({
                "config": config,
                "results": results,
            }, indent=2), encoding="utf-8",
        )

    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
