import itertools
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)

INPUTS = list(itertools.product((0, 1), repeat=3))
QUESTIONS = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
UNASKED = [x for x in INPUTS if x not in QUESTIONS]
TEST_RULES = [6, 9, 14, 15]
TRAIN_RULES = [r for r in range(16) if r not in TEST_RULES]

def environment(rule_ids, x):
    # Cette fonction appartient a l'environnement.
    # Le modele ne recoit jamais les identifiants des regles.
    shifts = torch.arange(4)
    coefficients = ((rule_ids[:, None] >> shifts) & 1)
    return (
        coefficients[:, 0, None]
        + (coefficients[:, None, 1:] * x).sum(-1)
    ) % 2

class MetaSSM(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, width), nn.Tanh())
        self.decay = nn.Parameter(torch.zeros(width))
        self.decoder = nn.Sequential(
            nn.Linear(width + 3, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, observed_x, observed_y, queries):
        batch = queries.size(0)
        h = torch.zeros(batch, self.decay.numel())
        decay = self.decay.sigmoid()

        for position in range(observed_x.size(1)):
            observation = torch.cat([
                observed_x[:, position].float() * 2 - 1,
                observed_y[:, position, None].float() * 2 - 1,
            ], dim=-1)
            h = decay * h + self.encoder(observation)

        state = h[:, None, :].expand(-1, queries.size(1), -1)
        return self.decoder(torch.cat([
            state, queries.float() * 2 - 1
        ], dim=-1))

@torch.no_grad()
def evaluate(model, rule_ids):
    model.eval()
    correct_before = correct_after = correct_observed = 0
    details = []
    orders = list(itertools.permutations(range(4)))

    for rule in rule_ids:
        rules = torch.tensor([rule])
        queries = torch.tensor([UNASKED])
        expected = environment(rules, queries)
        total_for_rule = before_for_rule = after_for_rule = 0

        for order in orders:
            observed_x = torch.tensor([[
                QUESTIONS[i] for i in order
            ]])
            observed_y = environment(rules, observed_x)

            before = model(
                observed_x[:, :0], observed_y[:, :0], queries
            ).argmax(-1)
            after = model(
                observed_x, observed_y, queries
            ).argmax(-1)
            seen = model(
                observed_x, observed_y, observed_x
            ).argmax(-1)

            b = int((before == expected).sum())
            a = int((after == expected).sum())
            correct_before += b
            correct_after += a
            correct_observed += int((seen == observed_y).sum())
            before_for_rule += b
            after_for_rule += a
            total_for_rule += 4

        details.append({
            "rule_id": rule,
            "correct_before": before_for_rule,
            "correct_after": after_for_rule,
            "total": total_for_rule,
        })

    total = len(rule_ids) * len(orders) * 4
    return {
        "correct_before": correct_before,
        "correct_after": correct_after,
        "correct_observed": correct_observed,
        "total": total,
        "per_rule": details,
    }

def main():
    output = Path("autonomous_lab/runs/meta_ssm_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)
    results = []

    for seed in (41001, 41002, 41003):
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed + 100)
        model = MetaSSM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_ids = torch.tensor(TRAIN_RULES)

        for step in range(1500):
            ids = train_ids[torch.randint(
                len(train_ids), (32,), generator=generator
            )]
            # Un ordre aleatoire par episode.
            orders = torch.rand(
                32, 4, generator=generator
            ).argsort(dim=1)
            observed_x = torch.tensor(QUESTIONS)[orders]
            observed_y = environment(ids, observed_x)
            queries = torch.tensor(INPUTS)[None].expand(32, -1, -1)
            expected = environment(ids, queries)

            model.train()
            logits = model(observed_x, observed_y, queries)
            loss = F.cross_entropy(
                logits.reshape(-1, 2), expected.reshape(-1)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Perte non finie.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()

            if (step + 1) % 500 == 0:
                print(
                    f"Graine {seed} | {step+1}/1500 | "
                    f"perte={loss.item():.4f}",
                    flush=True,
                )

        train = evaluate(model, TRAIN_RULES)
        test = evaluate(model, TEST_RULES)
        results.append({"seed": seed, "train": train, "test": test})

        for label, scores in (("TRAIN", train), ("REGLES_RESERVEES", test)):
            n = scores["total"]
            print(
                f"{label} | graine {seed} | "
                f"sans_observations={scores['correct_before']}/{n} | "
                f"avec_observations={scores['correct_after']}/{n} | "
                f"reponses_observees={scores['correct_observed']}/{n}",
                flush=True,
            )

        torch.save(
            model.state_dict(), output / f"model_{seed}.pt"
        )

    config = {
        "train_rules": TRAIN_RULES,
        "test_rules": TEST_RULES,
        "seeds": [41001, 41002, 41003],
        "training_steps": 1500,
        "batch_size": 32,
        "evaluation_updates": 0,
        "evaluation_orders": 24,
        "torch_version": str(torch.__version__),
    }
    (output / "results.json").write_text(
        json.dumps({"config": config, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
