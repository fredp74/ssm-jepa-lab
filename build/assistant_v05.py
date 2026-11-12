import argparse
import hashlib
import json
import math
import random
import signal
import string
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import assistant_v04 as extension

app = extension.previous
core = extension.core
ALPHABET = string.ascii_lowercase + string.digits
SAVE_EVERY = 25


def record(payload, template):
    question = [
        f"Recopie uniquement le texte entre crochets : [{payload}].",
        f"Reponds seulement avec le contenu de [{payload}].",
        f"Copie sans commentaire : [{payload}].",
        f"Quel texte est entre crochets ? [{payload}] Reponds seulement avec ce texte.",
    ][template]
    return {"task": f"copie_L{len(payload)}", "messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": payload},
    ]}


def answers(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text("utf-8-sig").splitlines()
        if line.strip()
    ]
    return rows, {row["messages"][-1]["content"] for row in rows}


def fresh(rng, used, reserved, maximum):
    for _ in range(10000):
        length = rng.randint(2, maximum)
        payload = "".join(rng.choice(ALPHABET) for _ in range(length))
        if payload not in used and payload not in reserved:
            used.add(payload)
            return record(payload, rng.randrange(4))
    raise RuntimeError("Generation sans doublon epuisee")


def maximum_length(step, steps):
    return 3 if step < steps//4 else 5 if step < steps//2 else 8


def prepare(path):
    if path.exists():
        raise FileExistsError("Test deja present : aucun remplacement")
    excluded = set()
    for name in ("simple_train", "simple_test"):
        source = Path("instructions_v03")/f"{name}.jsonl"
        _, values = answers(source)
        excluded.update(values)

    rng = random.Random(51005)
    rows = []
    for length in range(2, 9):
        for index in range(16):
            while True:
                payload = "".join(
                    rng.choice(ALPHABET) for _ in range(length)
                )
                if payload not in excluded:
                    break
            excluded.add(payload)
            rows.append(record(payload, index % 4))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
        encoding="utf-8"
    )
    print(
        f"Test fixe : {path} | 112 exemples | "
        "16 par longueur, de 2 a 8 | distinct des anciens exemples"
    )


def audit_add(state, row):
    payload = row["messages"][-1]["content"]
    bucket = state["stream_audit"].setdefault(str(len(payload)), [])
    rank = hashlib.sha256(payload.encode()).hexdigest()
    bucket.append({"rank": rank, "record": row})
    bucket.sort(key=lambda item: item["rank"])
    del bucket[16:]


def save(model, opt, state, rng, replay_rng, used, path):
    state["model"] = model.state_dict()
    state["sft_optimizer"] = opt.state_dict()
    state["stream_rng"] = rng.getstate()
    state["sft_rng"] = replay_rng.get_state()
    state["stream_used"] = sorted(used)
    state["torch_rng"] = torch.get_rng_state()
    core.atomic_save(state, path)

    rows = [
        item["record"]
        for length in sorted(state["stream_audit"], key=int)
        for item in state["stream_audit"][length]
    ]
    audit = path.with_suffix(".train_audit.jsonl")
    temporary = audit.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
        encoding="utf-8"
    )
    temporary.replace(audit)


def update(model, opt, state, rng, replay_rng, used, reserved):
    step = state["sft"]["step"]
    config = state["sft"]["config"]
    maximum = maximum_length(step, config["steps"])
    records = [fresh(rng, used, reserved, maximum) for _ in range(4)]
    rows = [app.encoded(row) for row in records]
    x, y = app.instruction_batch(rows, list(range(4)))
    content, ending = app.split_instruction_loss(model(x), y)

    indices = torch.randint(
        len(state["memory"]), (2,), generator=replay_rng
    )
    replay = state["memory"][indices].long()
    lm = F.cross_entropy(
        model(replay[:, :-1]).reshape(-1, 256),
        replay[:, 1:].reshape(-1)
    )
    loss = (
        (1-config["replay_weight"])*(0.9*content+0.1*ending)
        + config["replay_weight"]*lm
    )
    if not torch.isfinite(loss).item():
        raise RuntimeError("Perte non finie")

    lr = config["lr"]*min(1.0, (step+1)/20)*(
        0.1+0.45*(1+math.cos(math.pi*step/config["steps"]))
    )
    for group in opt.param_groups:
        group["lr"] = lr
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    opt.step()

    for row in records:
        audit_add(state, row)
    state["sft"]["step"] += 1
    return maximum, [
        value.item()/math.log(2) for value in (content, ending, lm)
    ]


def train(args):
    if args.base.resolve() == args.run.resolve():
        raise ValueError("Le run doit etre distinct du checkpoint de base")
    records, reserved = answers(args.test)
    if not records or not reserved:
        raise ValueError("Test vide")

    signature = {
        "objective": "fresh_copy_curriculum_v1",
        "steps": args.steps,
        "seed": args.seed,
        "lr": args.lr,
        "replay_weight": args.replay_weight,
        "base_sha256": core.fingerprint(args.base),
        "test_sha256": core.fingerprint(args.test),
        "trainer_sha256": core.fingerprint(__file__),
        "torch_version": str(torch.__version__),
    }

    if args.run.exists():
        if not args.resume:
            raise FileExistsError(
                "Run existant : utiliser --resume avec les memes arguments"
            )
        state = core.load(args.run)
        if state["sft"]["config"] != signature:
            raise ValueError(
                "Configuration ou fichiers modifies : reprise refusee"
            )
    else:
        if args.resume:
            raise FileNotFoundError("Run a reprendre absent")
        state = core.load(args.base)
        if state["arch"] != extension.ARCH or state.get("sft"):
            raise ValueError(
                "Partir de hybrid_base.pt avant les instructions"
            )
        phase = state.get("phase")
        if phase and phase["step"] < phase["steps"]:
            raise ValueError("Phase LM encore en cours")
        state["sft"] = {
            "step": 0, "config": signature, "elapsed_s": 0.0
        }
        state["sft_optimizer"] = None
        state["stream_rng"] = random.Random(args.seed).getstate()
        state["sft_rng"] = torch.Generator().manual_seed(
            args.seed+100_000
        ).get_state()
        state["stream_used"] = []
        state["stream_audit"] = {}

    model = extension.factory(state).train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    if state["sft_optimizer"] is not None:
        opt.load_state_dict(state["sft_optimizer"])
    torch.set_rng_state(state["torch_rng"])

    rng = random.Random()
    rng.setstate(state["stream_rng"])
    replay_rng = torch.Generator().set_state(state["sft_rng"])
    used = set(state["stream_used"])
    if used & reserved:
        raise ValueError("Collision train/test dans le checkpoint")

    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    previous = signal.signal(signal.SIGINT, stop)
    try:
        save(model, opt, state, rng, replay_rng, used, args.run)
        while state["sft"]["step"] < args.steps and not stopped:
            start = time.perf_counter()
            maximum, losses = update(
                model, opt, state, rng, replay_rng, used, reserved
            )
            state["sft"]["elapsed_s"] += time.perf_counter()-start
            step = state["sft"]["step"]
            if step % SAVE_EVERY == 0:
                save(model, opt, state, rng, replay_rng, used, args.run)
                minutes = state["sft"]["elapsed_s"]/60
                print(
                    f"STREAM {step}/{args.steps} | longueur 2..{maximum} | "
                    f"contenu {losses[0]:.3f} | fin {losses[1]:.3f} | "
                    f"replay {losses[2]:.3f} bpb | "
                    f"{len(used)} chaines distinctes | {minutes:.1f} min",
                    flush=True
                )
        save(model, opt, state, rng, replay_rng, used, args.run)
        print(
            f"Sauvegarde : {state['sft']['step']}/{args.steps} -> {args.run}"
        )
        print(f"Audit train : {args.run.with_suffix('.train_audit.jsonl')}")
    finally:
        signal.signal(signal.SIGINT, previous)


def smoke():
    rng = random.Random(4)
    used, reserved = set(), {"aa", "bb", "12"}
    rows = [fresh(rng, used, reserved, 3) for _ in range(80)]
    assert len(used) == 80 and not used & reserved
    assert all(
        2 <= len(row["messages"][-1]["content"]) <= 3 for row in rows
    )
    saved = rng.getstate()
    other = random.Random()
    other.setstate(saved)
    copied = set(used)
    assert fresh(rng, used, reserved, 8) == fresh(
        other, copied, reserved, 8
    )
    assert maximum_length(0, 1200) == 3
    assert maximum_length(300, 1200) == 5
    assert maximum_length(600, 1200) == 8
    extension.smoke()
    print(
        "STREAM SMOKE OK : exemples distincts, exclusion test, RNG et curriculum."
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")

    p = sub.add_parser("prepare")
    p.add_argument(
        "--out", type=Path,
        default=Path("instructions_v05/test.jsonl")
    )

    p = sub.add_parser("train")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--test", type=Path,
        default=Path("instructions_v05/test.jsonl")
    )
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--replay-weight", type=float, default=0.3)
    p.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out)
    elif args.command == "smoke":
        smoke()
    else:
        if (
            args.steps < 4 or args.lr <= 0
            or not 0 <= args.replay_weight < 1
        ):
            parser.error("Parametres invalides")
        train(args)


if __name__ == "__main__":
    main()
