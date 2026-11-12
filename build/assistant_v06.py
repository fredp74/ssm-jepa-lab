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


def record(payload, template, task="copie", rng=None, reserved=None):
    values = [payload]
    if task == "copie":
        question = [
            f"Recopie uniquement le texte entre crochets : [{payload}].",
            f"Reponds seulement avec le contenu de [{payload}].",
            f"Copie sans commentaire : [{payload}].",
            f"Quel texte est entre crochets ? [{payload}] Reponds seulement avec ce texte.",
        ][template]
    else:
        if task != "extraction" or rng is None:
            raise ValueError("Tache ou generateur invalide")
        reserved = set() if reserved is None else reserved
        for _ in range(2):
            for attempt in range(10000):
                word = "".join(
                    rng.choice(ALPHABET)
                    for _ in range(rng.randint(2, 5))
                )
                if word not in reserved and word not in values:
                    values.append(word)
                    break
            else:
                raise RuntimeError("Generation des distracteurs epuisee")
        fields = [
            f"cible={payload}",
            f"gauche={values[1]}",
            f"droite={values[2]}",
        ]
        rng.shuffle(fields)
        text = " ; ".join(fields)
        question = [
            f"Extrais uniquement la valeur du champ cible : {text}.",
            f"Donne seulement la valeur de cible dans : {text}.",
            f"Lis ces champs : {text}. Reponds uniquement avec la valeur de cible.",
            f"Quelle est la valeur de cible ? {text}. Sans commentaire.",
        ][template]
    return {
        "task": f"{task}_L{len(payload)}",
        "source_values": values,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": payload},
        ]
    }


def answers(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text("utf-8-sig").splitlines()
        if line.strip()
    ]
    values = {
        value for row in rows
        for value in row.get(
            "source_values", [row["messages"][-1]["content"]]
        )
    }
    return rows, values


def fresh(rng, used, reserved, maximum, task="copie"):
    for _ in range(10000):
        payload = "".join(
            rng.choice(ALPHABET)
            for _ in range(rng.randint(2, maximum))
        )
        if payload not in used and payload not in reserved:
            used.add(payload)
            return record(payload, rng.randrange(4), task, rng, reserved)
    raise RuntimeError("Generation sans doublon epuisee")


def maximum_length(step, steps):
    return 8


def prepare(path, base):
    if path.exists():
        raise FileExistsError("Test deja present : aucun remplacement")
    state = core.load(base)
    if "stream_used" not in state:
        raise ValueError("Utiliser assistant_stream_v05.pt termine")
    phase = state.get("sft")
    if not phase or phase["step"] < phase["config"]["steps"]:
        raise ValueError("Terminer la phase v0.5 avant cette preparation")

    _, old_test = answers(Path("instructions_v05/test.jsonl"))
    excluded = set(state["stream_used"]) | old_test
    rng = random.Random(61006)
    rows = []

    for task, maximum in (("copie", 8), ("extraction", 5)):
        for length in range(2, maximum+1):
            for index in range(16):
                for attempt in range(10000):
                    payload = "".join(
                        rng.choice(ALPHABET) for _ in range(length)
                    )
                    if payload not in excluded:
                        break
                else:
                    raise RuntimeError("Generation du test epuisee")
                row = record(payload, index % 4, task, rng, excluded)
                excluded.update(row["source_values"])
                rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
        encoding="utf-8"
    )
    print(
        f"Test fixe : {path} | 112 copies + 64 extractions | "
        "16 exemples par tache/longueur"
    )


def audit_add(state, row):
    payload = row["messages"][-1]["content"]
    bucket = state["stream_audit"].setdefault(row["task"], [])
    bucket.append({
        "rank": hashlib.sha256(payload.encode()).hexdigest(),
        "record": row
    })
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
        for length in sorted(state["stream_audit"])
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
    records = ([fresh(rng, used, reserved, 8, "copie") for _ in range(2)]
               + [fresh(rng, used, reserved, 5, "extraction") for _ in range(2)])
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
    _, prior_test = answers(Path("instructions_v05/test.jsonl"))
    reserved |= prior_test
    if not records or not reserved:
        raise ValueError("Test vide")

    signature = {
        "objective": "fresh_copy_extraction_v1",
        "steps": args.steps,
        "seed": args.seed,
        "lr": args.lr,
        "replay_weight": args.replay_weight,
        "base_sha256": core.fingerprint(args.base),
        "test_sha256": core.fingerprint(args.test),
        "prior_test_sha256": core.fingerprint(Path("instructions_v05/test.jsonl")),
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
        if state["arch"] != extension.ARCH or (state.get("sft") and state["sft"]["step"] < state["sft"]["config"]["steps"]):
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
        state["stream_used"] = list(state.get("stream_used", []))
        state["stream_initial_count"] = len(state["stream_used"])
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
                    f"STREAM {step}/{args.steps} | copie 2..8, extraction 2..5 | "
                    f"contenu {losses[0]:.3f} | fin {losses[1]:.3f} | "
                    f"replay {losses[2]:.3f} bpb | "
                    f"{len(used)-state['stream_initial_count']} nouvelles cibles | {minutes:.1f} min",
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
    import re
    rng = random.Random(6)
    used, reserved = set(), {"aa", "bb", "12"}
    rows = [
        fresh(rng, used, reserved, 5, "extraction")
        for _ in range(40)
    ]
    assert len(used) == 40 and not used & reserved
    positions = set()
    for row in rows:
        question = row["messages"][0]["content"]
        expected = row["messages"][1]["content"]
        assert re.search(
            r"cible=([a-z0-9]+)", question
        ).group(1) == expected
        assert len(set(row["source_values"])) == 3
        assert not set(row["source_values"]) & reserved
        positions.add(
            re.findall(r"(cible|gauche|droite)=", question).index("cible")
        )
    assert positions == {0, 1, 2}
    saved = rng.getstate()
    other = random.Random()
    other.setstate(saved)
    copied = set(used)
    assert fresh(
        rng, used, reserved, 5, "extraction"
    ) == fresh(other, copied, reserved, 5, "extraction")
    extension.smoke()
    print(
        "MIXED SMOKE OK : champs, distracteurs, positions variables, "
        "exclusion test et RNG."
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")

    p = sub.add_parser("prepare")
    p.add_argument("--base", type=Path, default=Path("runs/assistant_stream_v05.pt"))
    p.add_argument(
        "--out", type=Path,
        default=Path("instructions_v06/test.jsonl")
    )

    p = sub.add_parser("train")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--test", type=Path,
        default=Path("instructions_v06/test.jsonl")
    )
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--replay-weight", type=float, default=0.3)
    p.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.base)
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
