import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)

def encode_step(step, actions):
    features = torch.zeros(actions + 48 + 3)
    tested = step["tested_ids_before"]
    if tested:
        features[tested[-1]] = 1
    for candidate in step["known_candidates_before"]:
        features[actions + candidate] = 1
    features[-3:] = torch.tensor([
        step["remaining_calls"]/8,
        step["remaining_cells"]/24,
        step["remaining_text_bytes"]/512,
    ])
    return features

class PolicySSM(nn.Module):
    def __init__(self, actions, width=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(actions + 48 + 3, width), nn.Tanh()
        )
        self.decay = nn.Parameter(
            torch.full((width,), math.log(0.9/0.1))
        )
        self.head = nn.Linear(width, actions)

    def forward(self, x, lengths):
        h = torch.zeros(x.size(0), self.decay.numel(), device=x.device)
        outputs = []
        decay = self.decay.sigmoid()
        for position in range(x.size(1)):
            updated = decay*h + self.encoder(x[:, position])
            h = torch.where(
                (position < lengths)[:, None], updated, h
            )
            outputs.append(self.head(h))
        return torch.stack(outputs, dim=1)

class SSMPolicy:
    def __init__(self, model, actions):
        self.model = model.eval()
        self.actions = actions
        self.reset()

    def reset(self):
        self.history = []

    @torch.inference_mode()
    def choose(self, step, catalog):
        feasible = step["feasible_actions"]
        if not feasible:
            return None, "no_feasible_action"
        self.history.append(encode_step(step, self.actions))
        x = torch.stack(self.history)[None]
        logits = self.model(
            x, torch.tensor([len(self.history)])
        )[0, -1]
        mask = torch.zeros(self.actions, dtype=torch.bool)
        mask[feasible] = True
        action = int(logits.masked_fill(~mask, -1e9).argmax())
        return action, "ssm"

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=True
        )
        model = PolicySSM(checkpoint["actions"])
        model.load_state_dict(checkpoint["model"])
        return cls(model, checkpoint["actions"])

def read(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def pack(rows, actions):
    length = max(len(row["steps"]) for row in rows)
    x = torch.zeros(len(rows), length, actions+51)
    feasible = torch.zeros(len(rows), length, actions, dtype=torch.bool)
    target = torch.zeros(len(rows), length, actions)
    lengths = torch.tensor([len(row["steps"]) for row in rows])

    for i, row in enumerate(rows):
        for j, step in enumerate(row["steps"]):
            x[i, j] = encode_step(step, actions)
            feasible[i, j, step["feasible_actions"]] = True
            best = step["teacher_best_actions"]
            target[i, j, best] = 1/len(best)
    valid = torch.arange(length)[None] < lengths[:, None]
    return x, lengths, feasible, target, valid

@torch.inference_mode()
def evaluate(model, dataset):
    model.eval()
    x, lengths, feasible, target, valid = dataset
    logits = model(x, lengths).masked_fill(~feasible, -1e9)
    predictions = logits.argmax(-1)
    matches = target.gather(-1, predictions[:, :, None]).squeeze(-1) > 0
    return {
        "decisions": int(valid.sum()),
        "teacher_matches": int((matches & valid).sum()),
        "initial_decisions": len(lengths),
        "initial_matches": int(matches[:, 0].sum()),
        "later_decisions": int(valid.sum())-len(lengths),
        "later_matches": int((matches[:, 1:] & valid[:, 1:]).sum()),
        "infeasible_predictions": int((
            ~feasible.gather(-1, predictions[:, :, None]).squeeze(-1)
            & valid
        ).sum()),
    }

def main():
    root = Path("autonomous_lab")
    data = root / "runs/csv_policy_dataset_01"
    output = root / "runs/csv_ssm_policy_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    catalog = json.loads(
        (root / "data/csv_ssm_policy_01/catalog.json").read_text(
            encoding="utf-8"
        )
    )
    actions = len(catalog)
    train = pack(read(data / "train.jsonl"), actions)
    validation = pack(read(data / "validation.jsonl"), actions)

    config = {
        "seeds": [97001, 97002, 97003],
        "steps": 2000,
        "batch_size": 64,
        "learning_rate": 0.001,
        "width": 64,
        "initial_decay": 0.9,
        "actions": actions,
        "training": "teacher_forced_trajectory_imitation",
        "target": "uniform_over_tied_teacher_best_actions",
        "validation_states": "all_shared_with_train",
        "reserved_compositions_used": False,
        "checkpoint_selection": "final_step_only",
        "source_sha256": {
            name: hashlib.sha256((data/name).read_bytes()).hexdigest()
            for name in ("train.jsonl", "validation.jsonl")
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
        model = PolicySSM(actions)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x, lengths, feasible, targets, valid = train
        for step in range(2000):
            indices = torch.randint(len(x), (64,), generator=rng)
            model.train()
            logits = model(x[indices], lengths[indices])
            logits = logits.masked_fill(~feasible[indices], -1e9)
            losses = -(
                targets[indices]*F.log_softmax(logits, dim=-1)
            ).sum(-1)
            loss = losses[valid[indices]].mean()
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
                    f"Graine {seed} | {step+1}/2000 | perte={loss.item():.4f}",
                    flush=True,
                )

        result = {
            "seed": seed,
            "train": evaluate(model, train),
            "validation": evaluate(model, validation),
        }
        results.append(result)

        torch.save({
            "model": model.state_dict(),
            "actions": actions,
        }, output / f"model_{seed}.pt")

        print(f"BILAN {seed} : {result}", flush=True)
        (output / "results.json").write_text(
            json.dumps({
                "config": config,
                "results": results,
            }, indent=2), encoding="utf-8",
        )

    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
