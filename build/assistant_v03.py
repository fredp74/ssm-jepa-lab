"""GRU stateful inference and byte-level supervised instruction tuning."""
import argparse
import copy
import json
import math
import signal
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import continual as core

SYSTEM = "Tu es un assistant. Reponds a la consigne.\n"
USER, ASSISTANT, END = "<|user|>\n", "<|assistant|>\n", "\n<|end|>\n"
SAVE_EVERY = 25


def gru_model(state):
    if state["arch"] != "GRU":
        raise ValueError("v0.2 : generation avec etat et SFT disponibles pour GRU uniquement")
    return core.model_for(state)


def encoded(record):
    text = bytearray(SYSTEM.encode())
    mask = [False]*len(text)
    messages = record["messages"]
    if not messages or len(messages) % 2:
        raise ValueError("messages doit contenir des paires user/assistant")
    for i, message in enumerate(messages):
        role = "user" if i % 2 == 0 else "assistant"
        if message["role"] != role or not isinstance(message["content"], str) or not message["content"]:
            raise ValueError("Roles alternes user/assistant et contenus non vides attendus")
        if any(token in message["content"] for token in (USER.strip(), ASSISTANT.strip(), END.strip())):
            raise ValueError("Un contenu contient un marqueur reserve")
        prefix = USER if role == "user" else ASSISTANT
        suffix = "\n" if role == "user" else END
        part = (prefix+message["content"]+suffix).encode("utf-8")
        text.extend(part)
        mask.extend([False]*len(prefix.encode()))
        mask.extend([role == "assistant"]*(len(part)-len(prefix.encode())))
    return torch.tensor(list(text), dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def examples(path, maximum=512):
    rows = []
    for line, text in enumerate(Path(path).read_text("utf-8-sig").splitlines(), 1):
        if not text.strip():
            continue
        item = encoded(json.loads(text))
        if len(item[0]) > maximum:
            raise ValueError(f"Ligne {line} : {len(item[0])} octets > {maximum} ; aucune troncature appliquee")
        rows.append(item)
    if not rows:
        raise ValueError("Jeu d'instructions vide")
    return rows


def instruction_batch(rows, indices):
    selected = [rows[i] for i in indices]
    length = max(len(ids) for ids, _ in selected)-1
    x = torch.zeros((len(selected), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (ids, mask) in enumerate(selected):
        n = len(ids)-1
        x[i, :n] = ids[:-1]
        y[i, :n] = ids[1:].masked_fill(~mask[1:], -100)
    return x, y


@torch.no_grad()
def stream_bytes(model, prompt, count, temperature, topk, seed, stop=None):
    """Encode the whole prefix once, then carry GRU state byte by byte."""
    if not prompt:
        raise ValueError("Prefixe vide")
    model.eval()
    hidden = None
    for start in range(0, len(prompt), 256):
        x = torch.tensor([list(prompt[start:start+256])], dtype=torch.long)
        output, hidden = model.g(model.emb(x), hidden)
    logits = model.head(model.norm(output[:, -1]))[0]
    generator = torch.Generator().manual_seed(seed)
    result = bytearray()
    for _ in range(count):
        if temperature == 0:
            byte = logits.argmax().item()
        else:
            values, indices = (logits/temperature).topk(topk)
            j = torch.multinomial(values.softmax(0), 1, generator=generator)
            byte = indices[j].item()
        result.append(byte)
        if stop and result.endswith(stop):
            return bytes(result[:-len(stop)]), True
        x = torch.tensor([[byte]], dtype=torch.long)
        output, hidden = model.g(model.emb(x), hidden)
        logits = model.head(model.norm(output[:, -1]))[0]
    return bytes(result), False


def conversation_prefix(history, question, maximum):
    if not question.strip():
        raise ValueError("Question vide")
    for token in (USER.strip(), ASSISTANT.strip(), END.strip()):
        if token in question:
            raise ValueError("Marqueur reserve dans la question")
    history = list(history)
    while True:
        text = SYSTEM
        for user, answer in history:
            text += USER+user+"\n"+ASSISTANT+answer+END
        text += USER+question+"\n"+ASSISTANT
        if len(text.encode()) <= maximum:
            return text.encode(), history
        if not history:
            raise ValueError("Question trop longue pour le budget de contexte")
        history.pop(0)


def checkpoint(model, optimizer, state, generator, path):
    state["model"] = model.state_dict()
    state["sft_optimizer"] = optimizer.state_dict()
    state["sft_rng"] = generator.get_state()
    state["torch_rng"] = torch.get_rng_state()
    core.atomic_save(state, path)


def fit(args):
    if args.run.resolve() == args.base.resolve():
        raise ValueError("Le checkpoint SFT doit etre distinct de la base")
    rows = examples(args.data, args.max_bytes)
    signature = {
        "base_sha256": core.fingerprint(args.base),
        "data_sha256": core.fingerprint(args.data),
        "steps": args.steps, "seed": args.seed, "max_bytes": args.max_bytes,
        "lr": args.lr, "replay_weight": args.replay_weight,
        "format": 2, "instruction_loss": "0.9content+0.1end",
    }
    if args.run.exists():
        if not args.resume:
            raise FileExistsError("Run existant : utiliser --resume avec les memes arguments")
        state = core.load(args.run)
        if state["sft"]["config"] != signature:
            raise ValueError("Configuration ou fichiers modifies ; reprise refusee")
    else:
        if args.resume:
            raise FileNotFoundError("Run a reprendre introuvable")
        state = core.load(args.base)
        phase = state.get("phase")
        if phase and phase["step"] < phase["steps"]:
            raise ValueError("Terminer la phase LM avant le SFT")
        state["sft"] = {"step": 0, "config": signature}
        state["sft_optimizer"] = None
        state["sft_rng"] = torch.Generator().manual_seed(args.seed).get_state()
    torch.manual_seed(args.seed)
    model = gru_model(state).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if state["sft_optimizer"] is not None:
        opt.load_state_dict(state["sft_optimizer"])
    generator = torch.Generator().set_state(state["sft_rng"])
    torch.set_rng_state(state["torch_rng"])
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    previous = signal.signal(signal.SIGINT, stop)
    try:
        checkpoint(model, opt, state, generator, args.run)
        while state["sft"]["step"] < args.steps and not stopped:
            step = state["sft"]["step"]
            lr = args.lr*min(1.0, (step+1)/20)*(0.1+0.45*(1+math.cos(math.pi*step/args.steps)))
            for group in opt.param_groups:
                group["lr"] = lr
            indices = torch.randint(len(rows), (4,), generator=generator).tolist()
            x, y = instruction_batch(rows, indices)
            content_loss, end_loss = split_instruction_loss(model(x), y)
            instruction_loss = 0.9*content_loss + 0.1*end_loss
            indices = torch.randint(len(state["memory"]), (2,), generator=generator)
            replay = state["memory"][indices].long()
            replay_loss = F.cross_entropy(model(replay[:, :-1]).reshape(-1, 256), replay[:, 1:].reshape(-1))
            loss = (1-args.replay_weight)*instruction_loss+args.replay_weight*replay_loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Perte non finie")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            state["sft"]["step"] += 1
            if state["sft"]["step"] % SAVE_EVERY == 0:
                checkpoint(model, opt, state, generator, args.run)
                print(f"SFT {step+1}/{args.steps} | contenu {content_loss.item()/math.log(2):.3f} bpb | fin {end_loss.item()/math.log(2):.3f} bpb | replay {replay_loss.item()/math.log(2):.3f} bpb", flush=True)
        checkpoint(model, opt, state, generator, args.run)
        print(f"SFT sauvegarde : {state['sft']['step']}/{args.steps} -> {args.run}")
    finally:
        signal.signal(signal.SIGINT, previous)



def supervision_masks(y):
    marker = torch.tensor(list(END.encode()), dtype=torch.long)
    ending = torch.zeros_like(y, dtype=torch.bool)
    if y.size(1) >= len(marker):
        matches = y.unfold(1, len(marker), 1).eq(marker).all(-1)
        for row, start in matches.nonzero().tolist():
            ending[row, start:start+len(marker)] = True
    return (y != -100) & ~ending, ending


def split_instruction_loss(logits, y):
    losses = F.cross_entropy(
        logits.reshape(-1, 256), y.reshape(-1), reduction="none"
    ).reshape_as(y)
    content, ending = supervision_masks(y)
    if not content.any() or not ending.any():
        raise ValueError("Contenu ou marqueur de fin absent du batch")
    return losses[content].mean(), losses[ending].mean()


@torch.no_grad()
def evaluate(args):
    if args.limit_per_task < 1:
        raise ValueError("limit-per-task doit etre positif")
    model = gru_model(core.load(args.run)).eval()
    records = [
        json.loads(line)
        for line in args.data.read_text("utf-8-sig").splitlines()
        if line.strip()
    ]
    rows = examples(args.data)
    tasks = sorted({r.get("task", "autre") for r in records})

    for task in tasks:
        indices = [
            i for i, r in enumerate(records)
            if r.get("task", "autre") == task
        ]
        totals, counts = [0.0, 0.0], [0, 0]

        for start in range(0, len(indices), 4):
            x, y = instruction_batch(rows, indices[start:start+4])
            losses = F.cross_entropy(
                model(x).reshape(-1, 256),
                y.reshape(-1), reduction="none"
            ).reshape_as(y)
            for j, mask in enumerate(supervision_masks(y)):
                totals[j] += losses[mask].sum().item()
                counts[j] += mask.sum().item()

        bits = [
            totals[j]/max(counts[j], 1)/math.log(2)
            for j in range(2)
        ]
        print(
            f"PERTE {task} | n={len(indices)} | "
            f"contenu={bits[0]:.4f} bpb ({counts[0]} octets) | "
            f"fin={bits[1]:.4f} bpb ({counts[1]} octets)",
            flush=True
        )

        if not args.generate:
            continue

        gen = torch.Generator().manual_seed(42)
        order = torch.randperm(
            len(indices), generator=gen
        )[:args.limit_per_task].tolist()

        content_ok = eos_ok = joint_ok = dialogue_ok = 0
        for number, selected in enumerate(order):
            record = records[indices[selected]]
            history = []
            entire = True

            for i in range(0, len(record["messages"]), 2):
                question = record["messages"][i]["content"]
                expected = record["messages"][i+1]["content"]
                prefix, history = conversation_prefix(
                    history, question, 512
                )
                result, ended = stream_bytes(
                    model, prefix, 96, 0, 40, 42, END.encode()
                )
                answer = result.decode("utf-8", errors="replace")
                correct = answer.strip() == expected.strip()
                entire = entire and correct and ended
                history.append((question, answer))

            content_ok += int(correct)
            eos_ok += int(ended)
            joint_ok += int(correct and ended)
            dialogue_ok += int(entire)

            if number < 2:
                print(
                    f"EXEMPLE {task} | attendu={expected!r} | "
                    f"obtenu={answer!r} | EOS={ended}",
                    flush=True
                )

        n = len(order)
        print(
            f"GENERATIONS {task} | n={n} | "
            f"contenu_final={content_ok}/{n} | "
            f"EOS_final={eos_ok}/{n} | "
            f"exact_final+EOS={joint_ok}/{n} | "
            f"tous_les_tours_exacts+EOS={dialogue_ok}/{n}",
            flush=True
        )



def smoke():
    torch.manual_seed(4)
    model = core.GRU().eval()
    x = torch.randint(256, (1, 300))
    with torch.no_grad():
        whole = model(x)
        hidden = None
        pieces = []
        for i in range(0, 300, 37):
            output, hidden = model.g(model.emb(x[:, i:i+37]), hidden)
            pieces.append(model.head(model.norm(output)))
        torch.testing.assert_close(whole, torch.cat(pieces, 1), rtol=1e-4, atol=2e-6)
    example = {"messages": [{"role": "user", "content": "Dis oui."}, {"role": "assistant", "content": "oui"}]}
    row = encoded(example)
    a, y = instruction_batch([row], [0])
    supervised = bytes(y[0][y[0] != -100].tolist()).decode()
    assert supervised == "oui"+END
    loss = F.cross_entropy(model(a).reshape(-1, 256), y.reshape(-1))
    loss.backward()
    assert model.g.weight_ih_l0.grad is not None
    assert torch.isfinite(model.g.weight_ih_l0.grad).all()
    prefix, history = conversation_prefix([], "Dis oui.", 512)
    assert prefix.endswith(ASSISTANT.encode())
    print("SMOKE OK : GRU segmente equivalent, masque reponse/EOS, gradient SFT et format chat.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    for command in ("generate", "chat", "sft", "eval"):
        p = sub.add_parser(command)
        p.add_argument("--run", type=Path, required=True)
        if command in ("generate", "chat"):
            p.add_argument("--bytes", type=int, default=300)
            p.add_argument("--temperature", type=float, default=0.7)
            p.add_argument("--topk", type=int, default=40)
            p.add_argument("--seed", type=int, default=42)
        if command == "generate":
            p.add_argument("--prompt", required=True)
        if command == "chat":
            p.add_argument("--context-bytes", type=int, default=2048)
        if command in ("sft", "eval"):
            p.add_argument("--data", type=Path, required=True)
        if command == "eval":
            p.add_argument("--generate", action="store_true")
            p.add_argument("--limit-per-task", type=int, default=16)
        if command == "sft":
            p.add_argument("--base", type=Path, required=True)
            p.add_argument("--steps", type=int, default=400)
            p.add_argument("--seed", type=int, default=1000)
            p.add_argument("--max-bytes", type=int, default=512)
            p.add_argument("--lr", type=float, default=2e-4)
            p.add_argument("--replay-weight", type=float, default=0.3)
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cmd == "smoke":
        smoke()
        return
    if args.cmd == "sft":
        if args.steps < 1 or args.lr <= 0 or not 0 <= args.replay_weight < 1 or args.max_bytes < 2:
            parser.error("Parametres SFT invalides")
        fit(args)
        return
    if args.cmd == "eval":
        evaluate(args)
        return
    if args.bytes < 1 or args.temperature < 0 or not 1 <= args.topk <= 256:
        parser.error("Parametres de generation invalides")
    model = gru_model(core.load(args.run)).eval()
    if args.cmd == "generate":
        result, _ = stream_bytes(model, args.prompt.encode(), args.bytes, args.temperature, args.topk, args.seed)
        print(args.prompt+result.decode("utf-8", errors="replace"))
        return
    print("Chat local. /quit : quitter ; /reset : effacer l'historique. Aucun apprentissage automatique.")
    history = []
    while True:
        try:
            question = input("Vous > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question == "/quit":
            break
        if question == "/reset":
            history = []
            continue
        if not question.strip():
            continue
        try:
            prefix, retained = conversation_prefix(history, question, args.context_bytes)
            result, ended = stream_bytes(model, prefix, args.bytes, args.temperature, args.topk, args.seed+len(history), END.encode())
        except ValueError as error:
            print(error)
            continue
        answer = result.decode("utf-8", errors="replace")
        print("Assistant > "+answer)
        if ended:
            history = retained+[(question, answer)]
        else:
            print("[Limite d'octets atteinte ; reponse non ajoutee a l'historique.]")


if __name__ == "__main__":
    main()
