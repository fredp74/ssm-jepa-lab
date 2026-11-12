import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ssm_pilot import TinySSM, Environment, INPUTS, predictions

torch.set_num_threads(1)

QUESTIONS = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
HELDOUT = [x for x in INPUTS if x not in QUESTIONS]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 112),
            nn.Tanh(),
            nn.Linear(112, 2),
        )

    def forward(self, x):
        return self.network(x.float())

def main():
    output = Path("autonomous_lab/runs/informative_comparison_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    factories = {"ssm": TinySSM, "mlp": MLP}
    counts = {
        name: sum(p.numel() for p in factory().parameters())
        for name, factory in factories.items()
    }
    assert counts["ssm"] == counts["mlp"]
    print(f"PARAMETRES : {counts}", flush=True)

    results = []
    for seed in (31001, 31002, 31003):
        for architecture, factory in factories.items():
            for rule_id in range(16):
                torch.manual_seed(seed)
                model = factory()
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=0.01
                )
                environment = Environment(rule_id)
                expected = [environment.answer(x) for x in HELDOUT]
                before = predictions(model, HELDOUT)

                observed_x, observed_y = [], []
                for question in QUESTIONS:
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
                            model.parameters(), 1.0,
                            error_if_nonfinite=True,
                        )
                        optimizer.step()

                after = predictions(model, HELDOUT)
                seen = predictions(model, observed_x)
                results.append({
                    "architecture": architecture,
                    "seed": seed,
                    "rule_id": rule_id,
                    "questions": QUESTIONS,
                    "observed_answers": observed_y,
                    "heldout_inputs": HELDOUT,
                    "heldout_expected": expected,
                    "predictions_before": before,
                    "predictions_after": after,
                    "correct_before": sum(
                        a == b for a, b in zip(before, expected)
                    ),
                    "correct_after": sum(
                        a == b for a, b in zip(after, expected)
                    ),
                    "correct_seen": sum(
                        a == b for a, b in zip(seen, observed_y)
                    ),
                })

            rows = [
                r for r in results
                if r["seed"] == seed
                and r["architecture"] == architecture
            ]
            print(
                f"{architecture.upper()} | graine {seed} | "
                f"avant={sum(r['correct_before'] for r in rows)}/64 | "
                f"apres={sum(r['correct_after'] for r in rows)}/64 | "
                f"memorisees={sum(r['correct_seen'] for r in rows)}/64",
                flush=True,
            )

    config = {
        "parameters": counts,
        "seeds": [31001, 31002, 31003],
        "questions": QUESTIONS,
        "updates_per_question": 50,
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "reset_between_rules": True,
        "torch_version": str(torch.__version__),
    }
    path = output / "results.json"
    path.write_text(
        json.dumps({"config": config, "episodes": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
