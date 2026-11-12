import argparse
import copy
import functools
import hashlib
import json
import math
import signal
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(8)


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, 384)
        self.g = nn.GRU(384, 384, num_layers=2, batch_first=True)
        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, 256)

    def forward(self, x):
        return self.head(self.norm(self.g(self.emb(x))[0]))


class SSM(nn.Module):
    def __init__(self):
        super().__init__()
        import ssm_par
        from ssm_par import SSMEnc
        ssm_par.pscan = functools.partial(ssm_par.pscan, T=16)
        self.emb = nn.Embedding(256, 256)
        self.enc = SSMEnc(256, 256, depth=6, slow=True, par=True, n=4)
        self.head = nn.Linear(256, 256)

    def forward(self, x):
        return self.head(self.enc(self.emb(x)))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def model_for(state):
    model = GRU() if state["arch"] == "GRU" else SSM()
    model.load_state_dict(state["model"])
    return model


def data_for(state, domain, evaluation=False):
    root = Path(state["data_root"])
    if domain == "C":
        name = "dataC_test.pt" if evaluation else "dataC_train.pt"
        paths = [root/name]
        if not evaluation:
            paths.append(root/"dataC_manifest.json")
    else:
        name = f"data{domain}.pt"
        paths = [root/name]
    for path in paths:
        if state["hashes"][path.name] != fingerprint(path):
            raise ValueError(f"Fichier modifie : {path}")
    data = load(root/name)
    if domain == "C" and not evaluation:
        items = json.loads((root/"dataC_manifest.json").read_text("utf-8"))
        spans = [(int(i["offset"]), int(i["bytes"]))
                 for i in items if i["split"] == "train"]
    elif domain == "C":
        spans = [(0, len(data))]
    else:
        cut = len(data)-500_000
        spans = [(cut, 500_000)] if evaluation else [(0, cut)]
    if data.ndim != 1 or any(a < 0 or n <= state["length"] or a+n > len(data)
                             for a, n in spans) or not spans:
        raise ValueError("Donnees ou bornes incompatibles")
    return data, spans


def sample(data, spans, n, length, generator):
    counts = [size-length for _, size in spans]
    draws = torch.randint(sum(counts), (n,), generator=generator).tolist()
    starts = []
    for draw in draws:
        for (offset, _), count in zip(spans, counts):
            if draw < count:
                starts.append(offset+draw)
                break
            draw -= count
    return torch.stack([data[i:i+length+1] for i in starts]).long()


def advance(model, optimizer, state, data, spans):
    phase = state["phase"]
    main = torch.Generator().set_state(state["main_rng"])
    replay = torch.Generator().set_state(state["replay_rng"])
    selection = torch.Generator().set_state(state["selection_rng"])
    incoming = sample(data, spans, 8, state["length"], main)
    memory, protected = state["memory"], state["protected"]
    if protected:
        a = torch.randint(protected, (1,), generator=replay)
        b = torch.randint(protected, len(memory), (3,), generator=replay)
        indices = torch.cat([a, b])
    else:
        indices = torch.randint(len(memory), (4,), generator=replay)
    x = torch.cat([incoming, memory[indices].long()])
    step, steps = phase["step"], phase["steps"]
    lr = 1e-3*min(1.0, (step+1)/100)*(0.1+0.45*(1+math.cos(math.pi*step/steps)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss = F.cross_entropy(model(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
    if not torch.isfinite(loss).item():
        raise RuntimeError("Perte non finie")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    for sequence in incoming:
        state["seen"] += 1
        j = torch.randint(state["seen"], (1,), generator=selection).item()
        if j < len(memory)-protected:
            memory[protected+j].copy_(sequence.to(torch.uint8))
    phase["step"] += 1
    state["main_rng"] = main.get_state()
    state["replay_rng"] = replay.get_state()
    state["selection_rng"] = selection.get_state()
    return loss.item()/math.log(2)


def snapshot(model, optimizer, state, path):
    state["model"] = model.state_dict()
    state["optimizer"] = optimizer.state_dict()
    state["torch_rng"] = torch.get_rng_state()
    atomic_save(state, path)


def train(args):
    state = load(args.run)
    if state["version"] != 1:
        raise ValueError("Version incompatible")
    phase = state["phase"]
    fresh = args.cmd == "train"
    if fresh:
        if phase and phase["step"] < phase["steps"]:
            raise ValueError("Phase en cours : utiliser resume")
        state["phase"] = {"domain": args.domain, "step": 0, "steps": args.steps, "elapsed_s": 0.0}
        seed = args.seed
        state["main_rng"] = torch.Generator().manual_seed(seed).get_state()
        state["replay_rng"] = torch.Generator().manual_seed(seed+100_000).get_state()
        state["optimizer"] = None
    elif not phase or phase["step"] == phase["steps"]:
        print("Aucune phase interrompue a reprendre.")
        return
    data, spans = data_for(state, state["phase"]["domain"])
    model = model_for(state).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    if state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        snapshot(model, optimizer, state, args.run)
        while state["phase"]["step"] < state["phase"]["steps"]:
            if stopped:
                break
            started = time.perf_counter()
            loss = advance(model, optimizer, state, data, spans)
            state["phase"]["elapsed_s"] += time.perf_counter()-started
            step = state["phase"]["step"]
            if step % 25 == 0:
                snapshot(model, optimizer, state, args.run)
                print(f"{state['phase']['domain']} {step}/{state['phase']['steps']} | train {loss:.3f} bpb | {state['phase']['elapsed_s']/60:.1f} min", flush=True)
        snapshot(model, optimizer, state, args.run)
        print(f"Sauvegarde : pas {state['phase']['step']} -> {args.run}")
    finally:
        signal.signal(signal.SIGINT, previous)


def initialize(args):
    if args.run.exists():
        raise FileExistsError("Run deja present : choisir un autre nom")
    root = args.data.resolve()
    names = ["dataA.pt", "dataB.pt", "dataC_train.pt", "dataC_test.pt", "dataC_manifest.json"]
    torch.manual_seed(1000)
    model = GRU() if args.arch == "GRU" else SSM()
    checkpoint = root/("lm_GRU_A.pt" if args.arch == "GRU" else "lm_SSM-n4_A.pt")
    model.load_state_dict(load(checkpoint))
    data = load(root/"dataA.pt")
    generator = torch.Generator().manual_seed(424242)
    slots = (len(data)-500_000)//257
    if slots < 4096:
        raise ValueError("Train A trop petit")
    starts = torch.randperm(slots, generator=generator)[:4096]*257
    memory = torch.stack([data[i:i+257] for i in starts.tolist()]).to(torch.uint8)
    protected = 1024 if args.policy == "protected" else 0
    state = {
        "version": 1, "arch": args.arch, "data_root": str(root), "length": 256,
        "hashes": {name: fingerprint(root/name) for name in names},
        "model": model.state_dict(), "memory": memory, "protected": protected,
        "seen": len(memory)-protected, "phase": None, "optimizer": None,
        "selection_rng": torch.Generator().manual_seed(301000).get_state(),
        "torch_rng": torch.get_rng_state(), "history": [],
    }
    atomic_save(state, args.run)
    print(f"Initialise : {args.run} | {args.arch} | {args.policy}")


@torch.no_grad()
def evaluate(args):
    state = load(args.run)
    model = model_for(state).eval()
    scores = {}
    for domain in "ABC":
        data, spans = data_for(state, domain, evaluation=True)
        lo, size = spans[0]
        indices = torch.linspace(lo, lo+size-state["length"]-2, args.samples).long().tolist()
        total = 0.0
        for i in range(0, len(indices), 8):
            x = torch.stack([data[j:j+257] for j in indices[i:i+8]]).long()
            total += F.cross_entropy(model(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1), reduction="sum").item()
        scores[domain] = total/(len(indices)*256*math.log(2))
    row = {"phase": copy.deepcopy(state["phase"]), "samples": args.samples, "bpb": scores}
    state["history"].append(row)
    atomic_save(state, args.run)
    with args.run.with_suffix(".eval.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row)+"\n")
    print(json.dumps(row, ensure_ascii=False, indent=2))


@torch.no_grad()
def generate(args):
    state = load(args.run)
    model = model_for(state).eval()
    buffer = bytearray(args.prompt.encode("utf-8"))
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.bytes):
        x = torch.tensor([list(buffer[-256:])], dtype=torch.long)
        logits = model(x)[0, -1]/args.temperature
        values, indices = logits.topk(args.topk)
        choice = torch.multinomial(values.softmax(0), 1, generator=generator)
        buffer.append(indices[choice].item())
    print(buffer.decode("utf-8", errors="replace"))


def smoke():
    torch.manual_seed(42)
    model = GRU()
    data = torch.randint(256, (500,), dtype=torch.uint8)
    state = {"arch": "GRU", "length": 16, "protected": 1,
             "memory": torch.randint(256, (8, 17), dtype=torch.uint8), "seen": 7,
             "phase": {"domain": "B", "step": 0, "steps": 3}}
    for key, seed in [("main_rng", 1), ("replay_rng", 2), ("selection_rng", 3)]:
        state[key] = torch.Generator().manual_seed(seed).get_state()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    advance(model, opt, state, data, [(0, 500)])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/"smoke.pt"
        snapshot(model, opt, state, path)
        advance(model, opt, state, data, [(0, 500)])
        expected = {k: v.clone() for k, v in model.state_dict().items()}
        resumed = load(path)
        other = model_for(resumed)
        opt2 = torch.optim.AdamW(other.parameters(), lr=1e-3, weight_decay=0.01)
        opt2.load_state_dict(resumed["optimizer"])
        torch.set_rng_state(resumed["torch_rng"])
        advance(other, opt2, resumed, data, [(0, 500)])
        for k, value in other.state_dict().items():
            torch.testing.assert_close(value, expected[k], rtol=0, atol=0)
        assert torch.equal(resumed["memory"], state["memory"])
        assert resumed["seen"] == state["seen"]
        for k in ("main_rng", "replay_rng", "selection_rng"):
            assert torch.equal(resumed[k], state[k])
    print("SMOKE OK : reprise exacte sur 2 updates, optimiseur, memoire et RNG.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    for name in ("init", "train", "resume", "eval", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        if name == "init":
            p.add_argument("--data", type=Path, default=Path(".."))
            p.add_argument("--arch", choices=["GRU", "SSM"], default="GRU")
            p.add_argument("--policy", choices=["uniform", "protected"], default="protected")
        if name == "train":
            p.add_argument("--domain", choices=["A", "B", "C"], required=True)
            p.add_argument("--steps", type=int, default=2000)
            p.add_argument("--seed", type=int, default=1000)
        if name == "eval":
            p.add_argument("--samples", type=int, default=200)
        if name == "generate":
            p.add_argument("--prompt", required=True)
            p.add_argument("--bytes", type=int, default=200)
            p.add_argument("--temperature", type=float, default=0.8)
            p.add_argument("--topk", type=int, default=40)
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cmd == "generate" and (not args.prompt or args.temperature <= 0 or not 1 <= args.topk <= 256 or args.bytes < 0):
        parser.error("Parametres de generation invalides")
    if args.cmd == "train" and args.steps < 1:
        parser.error("steps doit etre positif")
    if args.cmd == "eval" and args.samples < 1:
        parser.error("samples doit etre positif")
    if args.cmd == "smoke":
        smoke()
    elif args.cmd == "init":
        initialize(args)
    elif args.cmd in ("train", "resume"):
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
