import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
import assistant_v04 as extension

app = extension.previous
core = extension.core

def read(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("selection", "mixed"), required=True)
    args = parser.parse_args()

    base = Path("runs/experiment_v13_control_130015.pt")
    destination = Path(f"runs/experiment_v20_{args.arm}.pt")
    selection_path = Path("instructions_v18/train.jsonl")
    mixed_path = Path("instructions_v19/mixed.jsonl")

    if destination.exists():
        raise SystemExit("Sauvegarde existante : aucun remplacement.")

    selection = read(selection_path)
    old = [
        row for row in read(mixed_path)
        if not row["task"].startswith("selection_")
    ]
    if len(selection) != 96 or len(old) != 96:
        raise SystemExit("Il faut 96 exemples dans chaque groupe.")

    selection = [app.encoded(row) for row in selection]
    old = [app.encoded(row) for row in old]

    state = core.load(base)
    torch.manual_seed(200020)
    model = extension.factory(state).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.01
    )

    # Generateurs separes : les exemples de selection et de replay
    # restent identiques entre les deux branches.
    selection_rng = torch.Generator().manual_seed(200021)
    replay_rng = torch.Generator().manual_seed(200022)
    old_rng = torch.Generator().manual_seed(200023)

    config = {
        "objective": "matched_selection_exposure_v20",
        "arm": args.arm,
        "steps": 400,
        "seed": 200020,
        "lr": 0.00005,
        "selection_weight": 0.7,
        "replay_weight": 0.3,
        "old_weight": 0.7 if args.arm == "mixed" else 0.0,
        "base_sha256": core.fingerprint(base),
        "selection_sha256": core.fingerprint(selection_path),
        "mixed_sha256": core.fingerprint(mixed_path),
        "trainer_sha256": core.fingerprint(__file__),
        "torch_version": str(torch.__version__),
    }
    state["sft"] = {"step": 0, "config": config}

    def instruction_loss(rows, generator):
        indices = torch.randint(
            len(rows), (4,), generator=generator
        ).tolist()
        x, y = app.instruction_batch(rows, indices)
        content, ending = app.split_instruction_loss(model(x), y)
        return 0.9 * content + 0.1 * ending

    def save():
        state["model"] = model.state_dict()
        state["sft_optimizer"] = optimizer.state_dict()
        state["torch_rng"] = torch.get_rng_state()
        state["v20_rng"] = {
            "selection": selection_rng.get_state(),
            "replay": replay_rng.get_state(),
            "old": old_rng.get_state(),
        }
        core.atomic_save(state, destination)

    save()
    try:
        for step in range(400):
            lr = 0.00005 * min(1.0, (step + 1) / 20) * (
                0.1 + 0.45 * (1 + math.cos(math.pi * step / 400))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            selection_loss = instruction_loss(selection, selection_rng)
            (0.7 * selection_loss).backward()

            old_value = 0.0
            if args.arm == "mixed":
                old_loss = instruction_loss(old, old_rng)
                old_value = old_loss.item()
                (0.7 * old_loss).backward()

            indices = torch.randint(
                len(state["memory"]), (2,), generator=replay_rng
            )
            replay = state["memory"][indices].long()
            replay_loss = F.cross_entropy(
                model(replay[:, :-1]).reshape(-1, 256),
                replay[:, 1:].reshape(-1),
            )
            (0.3 * replay_loss).backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            state["sft"]["step"] = step + 1

            if (step + 1) % 50 == 0:
                save()
                print(
                    f"{args.arm} {step+1}/400 | "
                    f"selection={selection_loss.item():.4f} | "
                    f"ancien={old_value:.4f} | "
                    f"replay={replay_loss.item():.4f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Interruption : sauvegarde du dernier pas termine.")
    finally:
        save()

    print(f"Sauvegarde : {state['sft']['step']}/400 -> {destination}")

if __name__ == "__main__":
    main()
