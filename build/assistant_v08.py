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


def single_record(payload, template, task="copie", rng=None, reserved=None):
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


def fresh(rng, used, reserved, maximum, task="copie", length=None):
    length = rng.randint(2, maximum) if length is None else length
    for _ in range(10000):
        payload = "".join(rng.choice(ALPHABET) for _ in range(length))
        if payload not in reserved:
            used.add(payload)
            return record(payload, rng.randrange(4), task, rng, reserved)
    raise RuntimeError("Generation hors test epuisee")



def maximum_length(step, steps):
    return 8


def prepare(path, base):
    if path.exists():
        raise FileExistsError("Test deja present : aucun remplacement")
    state = core.load(base)
    phase = state.get("sft")
    if not phase or phase["step"] < phase["config"]["steps"]:
        raise ValueError("Utiliser le checkpoint termine de la v0.7")
    excluded = set(state["stream_used"])
    for name in (
        "instructions_v05/test.jsonl",
        "instructions_v06/test.jsonl",
        "instructions_v07/fresh_validation.jsonl",
    ):
        _, values = answers(Path(name))
        excluded.update(values)

    rng = random.Random(81009)
    targets = []
    for task, maximum in (
        ("copie", 8), ("extraction", 5), ("dialogue", 4)
    ):
        for length in range(2, maximum+1):
            for index in range(16):
                for attempt in range(10000):
                    payload = "".join(
                        rng.choice(ALPHABET) for _ in range(length)
                    )
                    if payload not in excluded:
                        break
                else:
                    raise RuntimeError(
                        f"Pas assez de cibles de longueur {length}"
                    )
                excluded.add(payload)
                targets.append((task, payload, index))

    rows = []
    for task, payload, index in targets:
        row = record(
            payload, index % 4, task, rng, excluded, (index//4) % 4
        )
        excluded.update(row["source_values"])
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
        encoding="utf-8"
    )
    print(
        f"Test fixe : {path} | "
        "112 copies + 64 extractions + 48 dialogues"
    )



def audit_add(state, row):
    payload = row["messages"][-1]["content"]
    bucket = state["stream_audit"].setdefault(row["task"], [])
    if any(
        item["record"]["messages"][-1]["content"] == payload
        for item in bucket
    ):
        return
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
    records = [fresh(rng, used, reserved, 8, "copie", 2+step%7)]
    records += [fresh(rng, used, reserved, 5, "extraction", 2+step%4)]
    records += [fresh(rng, used, reserved, 4, "dialogue", 2+(2*step+j)%3) for j in range(2)]
    rows = [app.encoded(row) for row in records]
    x, y = app.instruction_batch(rows, list(range(4)))
    content, boundary, tail = loss3(model(x), y)

    indices = torch.randint(
        len(state["memory"]), (2,), generator=replay_rng
    )
    replay = state["memory"][indices].long()
    lm = F.cross_entropy(
        model(replay[:, :-1]).reshape(-1, 256),
        replay[:, 1:].reshape(-1)
    )
    instruction = 0.75*content+0.20*boundary+0.05*tail
    loss = (
        (1-config["replay_weight"])*instruction
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
        key = row["task"]
        state["balanced_counts"][key] = (
            state["balanced_counts"].get(key, 0)+1
        )
    state["sft"]["step"] += 1
    return 8, [
        value.item()/math.log(2)
        for value in (content, boundary, tail, lm)
    ]



def train(args):
    if args.base.resolve() == args.run.resolve():
        raise ValueError("Le run doit etre distinct du checkpoint de base")
    records, reserved = answers(args.test)
    for name in ("instructions_v06/test.jsonl", "instructions_v07/fresh_validation.jsonl"):
        _, values = answers(Path(name))
        reserved.update(values)
    _, prior_test = answers(Path("instructions_v05/test.jsonl"))
    reserved |= prior_test
    if not records or not reserved:
        raise ValueError("Test vide")

    signature = {
        "objective": "balanced_copy_extract_dialogue_v1",
        "steps": args.steps,
        "seed": args.seed,
        "lr": args.lr,
        "replay_weight": args.replay_weight,
        "base_sha256": core.fingerprint(args.base),
        "test_sha256": core.fingerprint(args.test),
        "prior_test_sha256": core.fingerprint(Path("instructions_v05/test.jsonl")),
        "additional_tests_sha256": {name: core.fingerprint(Path(name)) for name in ("instructions_v06/test.jsonl", "instructions_v07/fresh_validation.jsonl")},
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
        state["balanced_counts"] = {}

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
                    f"STREAM {step}/{args.steps} | copie/extraction/dialogue | "
                    f"contenu {losses[0]:.3f} | debut_fin {losses[1]:.3f} | suite_fin {losses[2]:.3f} | "
                    f"replay {losses[3]:.3f} bpb | "
                    f"{len(used)-state['stream_initial_count']} cibles distinctes supplementaires | {minutes:.1f} min",
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
    rng = random.Random(8)
    row = record("z9q", 0, "dialogue", rng)
    assert "z9q" in row["messages"][0]["content"]
    assert "z9q" not in row["messages"][2]["content"]
    assert row["messages"][1]["content"] == "OK"

    x, y = app.instruction_batch([app.encoded(row)], [0])
    content, boundary, tail = masks3(y)
    assert bytes(y[content].tolist()).decode() == "OKz9q"
    assert boundary.sum().item() == 2
    assert tail.sum().item() == 2*(len(app.END.encode())-1)

    extension.smoke()
    print(
        "DIALOGUE SMOKE OK : format deux tours, masques "
        "et code absent du dernier message."
    )




def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, default=Path("instructions_v08/test.jsonl"))
    p.add_argument("--base", type=Path, default=Path("runs/assistant_balanced_v07.pt"))

    p = sub.add_parser("train")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--test", type=Path,
        default=Path("instructions_v08/test.jsonl")
    )
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--replay-weight", type=float, default=0.3)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--data", type=Path,
        default=Path("instructions_v08/test.jsonl")
    )
    p.add_argument("--generate", action="store_true")
    p.add_argument("--limit-per-task", type=int, default=16)
    p.add_argument("--history-controls", action="store_true")

    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "prepare":
        prepare(args.out, args.base)
    elif args.command == "eval":
        boundary_report(args)
        app.evaluate(args)
        if args.history_controls:
            history_controls(args)
    else:
        if (
            args.steps < 4 or args.lr <= 0
            or not 0 <= args.replay_weight < 1
        ):
            parser.error("Parametres invalides")
        train(args)
        state = core.load(args.run)
        print(
            "EXEMPLES DE CETTE ETAPE : "
            + json.dumps(state["balanced_counts"], sort_keys=True)
        )



def masks3(y):
    content, ending = app.supervision_masks(y)
    previous = torch.cat([
        torch.zeros_like(ending[:, :1]), ending[:, :-1]
    ], dim=1)
    boundary = ending & ~previous
    return content, boundary, ending & ~boundary


def loss3(logits, y):
    losses = F.cross_entropy(
        logits.reshape(-1, 256), y.reshape(-1), reduction="none"
    ).reshape_as(y)
    masks = masks3(y)
    if not all(mask.any().item() for mask in masks):
        raise ValueError("Groupe de supervision vide")
    return tuple(losses[mask].mean() for mask in masks)


@torch.no_grad()
def boundary_report(args):
    model = extension.factory(core.load(args.run)).eval()
    records, _ = answers(args.data)
    rows = app.examples(args.data)
    for task in sorted({row["task"] for row in records}):
        indices = [
            i for i, row in enumerate(records) if row["task"] == task
        ]
        total = count = correct = 0
        for start in range(0, len(indices), 4):
            x, y = app.instruction_batch(rows, indices[start:start+4])
            logits = model(x)
            losses = F.cross_entropy(
                logits.reshape(-1, 256),
                y.reshape(-1), reduction="none"
            ).reshape_as(y)
            boundary = masks3(y)[1]
            total += losses[boundary].sum().item()
            count += boundary.sum().item()
            correct += (
                logits.argmax(-1)[boundary] == y[boundary]
            ).sum().item()
        print(
            f"DEBUT_FIN {task} | {total/count/math.log(2):.4f} bpb | "
            f"premier octet correct {correct}/{count} "
            "avec prefixe de reference",
            flush=True
        )


def record(payload, template, task="copie", rng=None, reserved=None, last_style=None):
    if task != "dialogue":
        return single_record(payload, template, task, rng, reserved)
    last_style = rng.randrange(4) if last_style is None else last_style
    first = [
        f"Retiens ce code : {payload}. Reponds seulement OK.",
        f"Memorise ce code : {payload}. Confirme uniquement par OK.",
        f"Le code a retenir est {payload}. Reponds uniquement OK.",
        f"Garde en memoire le code {payload}. Ecris seulement OK.",
    ][template]
    last = [
        "Quel code viens-je de donner ? Reponds seulement avec ce code.",
        "Redonne uniquement le code indique au message precedent.",
        "Quel est le code memorise ? Sans commentaire.",
        "Reponds seulement avec le code que je t'ai donne.",
    ][last_style]
    return {
        "task": f"dialogue_L{len(payload)}",
        "source_values": [payload],
        "first_style": template,
        "last_style": last_style,
        "messages": [
            {"role": "user", "content": first},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": last},
            {"role": "assistant", "content": payload},
        ]
    }


@torch.no_grad()
def history_controls(args):
    model = extension.factory(core.load(args.run)).eval()
    records, _ = answers(args.data)
    groups = {}
    for row in records:
        if row["task"].startswith("dialogue_"):
            key = (
                len(row["messages"][-1]["content"]),
                row["first_style"]
            )
            groups.setdefault(key, []).append(row)

    totals = {}
    for (length, style), rows in sorted(groups.items()):
        if len(rows) < 2:
            raise ValueError(
                "Au moins deux dialogues par longueur/style pour permuter"
            )
        stats = totals.setdefault(length, [0, 0, 0, 0, 0])
        for i, row in enumerate(rows):
            other = rows[(i+1) % len(rows)]
            question = row["messages"][2]["content"]
            expected = row["messages"][3]["content"]
            transplanted = other["messages"][3]["content"]
            histories = [
                [(row["messages"][0]["content"], "OK")],
                [],
                [(other["messages"][0]["content"], "OK")],
            ]
            outcomes = []
            for history in histories:
                prefix, _ = app.conversation_prefix(
                    history, question, 512
                )
                result, ended = app.stream_bytes(
                    model, prefix, 96, 0, 40, 42, app.END.encode()
                )
                outcomes.append((
                    result.decode("utf-8", errors="replace").strip(),
                    ended
                ))
            stats[0] += 1
            for j in range(3):
                stats[j+1] += int(
                    outcomes[j][1] and outcomes[j][0] == expected
                )
            stats[4] += int(
                outcomes[2][1] and outcomes[2][0] == transplanted
            )

    for length, stats in sorted(totals.items()):
        n, normal, absent, wrong_old, follows_new = stats
        print(
            f"HISTOIRE L{length} | n={n} | "
            f"reference_OK={normal}/{n} | "
            f"sans_histoire={absent}/{n} | "
            f"permute_repond_ancien={wrong_old}/{n} | "
            f"permute_suit_nouveau={follows_new}/{n}",
            flush=True
        )


if __name__ == "__main__":
    main()
