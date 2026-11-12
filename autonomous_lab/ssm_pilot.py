import itertools
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)

INPUTS = list(itertools.product((0, 1), repeat=3))

class Environment:
    def __init__(self, rule_id):
        self.coefficients = [(rule_id >> i) & 1 for i in range(4)]

    def answer(self, x):
        c = self.coefficients
        return (c[0] + sum(a*b for a, b in zip(c[1:], x))) % 2

class TinySSM(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.embedding = nn.Embedding(2, width)
        self.decay = nn.Parameter(torch.zeros(width))
        self.drive = nn.Parameter(torch.ones(width))
        self.head = nn.Sequential(
            nn.Linear(width, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        h = torch.zeros(x.size(0), self.decay.numel(), device=x.device)
        decay = self.decay.sigmoid()
        for position in range(x.size(1)):
            h = decay*h + self.drive*self.embedding(x[:, position])
        return self.head(h)

@torch.no_grad()
def predictions(model, inputs):
    model.eval()
    x = torch.tensor(inputs, dtype=torch.long)
    return model(x).argmax(-1).tolist()

def main():
    output = Path("autonomous_lab/runs/ssm_pilot_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    results = []
    for seed in (31001, 31002, 31003):
        for rule_id in range(16):
            # Meme initialisation et memes questions pour toutes les
            # regles d'une graine : seuls les retours changent.
            torch.manual_seed(seed)
            model = TinySSM()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            environment = Environment(rule_id)

            order = list(INPUTS)
            random.Random(seed).shuffle(order)
            questions, heldout = order[:4], order[4:]
            before = predictions(model, heldout)

            observed_x, observed_y = [], []
            for question in questions:
                observed_x.append(question)
                observed_y.append(environment.answer(question))

                x = torch.tensor(observed_x, dtype=torch.long)
                y = torch.tensor(observed_y, dtype=torch.long)
                model.train()
                for _ in range(50):
                    loss = F.cross_entropy(model(x), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Perte non finie.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()

            after = predictions(model, heldout)
            seen = predictions(model, observed_x)
            expected = [environment.answer(x) for x in heldout]

            row = {
                "seed": seed,
                "rule_id": rule_id,
                "questions": [list(x) for x in questions],
                "observed_answers": observed_y,
                "heldout_inputs": [list(x) for x in heldout],
                "heldout_expected": expected,
                "predictions_before": before,
                "predictions_after": after,
                "correct_before": sum(a == b for a, b in zip(before, expected)),
                "correct_after": sum(a == b for a, b in zip(after, expected)),
                "correct_seen": sum(a == b for a, b in zip(seen, observed_y)),
            }
            results.append(row)

        rows = [row for row in results if row["seed"] == seed]
        print(
            f"Graine {seed} | "
            f"avant={sum(r['correct_before'] for r in rows)}/64 | "
            f"apres={sum(r['correct_after'] for r in rows)}/64 | "
            f"questions_memorisees={sum(r['correct_seen'] for r in rows)}/64",
            flush=True,
        )

    config = {
        "architecture": "diagonal_ssm_nonlinear_readout",
        "width": 16,
        "questions_per_episode": 4,
        "updates_per_question": 50,
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "reset_between_rules": True,
        "question_strategy": "random_without_replacement",
        "torch_version": str(torch.__version__),
    }
    (output / "results.json").write_text(
        json.dumps({"config": config, "episodes": results}, indent=2),
        encoding="utf-8",
    )

    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
