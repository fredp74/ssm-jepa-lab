import argparse
import copy
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ssm_par import SSMEnc


class LanguageModel(nn.Module):
    def __init__(self, width=256, layers=6, state=4, ffn_mult=0):
        super().__init__()
        self.emb = nn.Embedding(256, width)
        self.enc = SSMEnc(
            width, width, depth=layers, slow=True, par=True, n=state
        )
        self.head = nn.Linear(width, 256)
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ffn_mult*width),
                          nn.GELU(), nn.Linear(ffn_mult*width, width))
            for _ in range(layers)
        ] if ffn_mult else [])

    def encode(self, tokens):
        x = self.enc.p(self.emb(tokens))
        for index, block in enumerate(self.enc.blocks):
            x = block(x)
            if self.ffns:
                x = x + self.ffns[index](x)
        return self.enc.norm(x)

    def forward(self, tokens):
        return self.head(self.encode(tokens))

    def step(self, token, states=None):
        x = self.enc.p(self.emb(token))
        updated = []
        for index, block in enumerate(self.enc.blocks):
            residual = x
            u, gate = block.inp(block.norm(x)).chunk(2, -1)
            u = F.silu(u)
            dt = F.softplus(block.dt(u))
            h = (
                torch.zeros(
                    u.size(0), block.d, block.n,
                    device=u.device, dtype=u.dtype,
                )
                if states is None else states[index]
            )
            decay = torch.exp(
                dt.unsqueeze(-1) * -torch.exp(block.A_log)
            )
            h = (
                decay * h
                + dt.unsqueeze(-1)
                * block.B(u).unsqueeze(1)
                * u.unsqueeze(-1)
            )
            y = (
                h * block.C(u).unsqueeze(1)
            ).sum(-1) + u * block.D
            x = residual + block.out(y * F.silu(gate))
            if self.ffns:
                x = x + self.ffns[index](x)
            updated.append(h)
        return self.head(self.enc.norm(x)), updated


def load(path):
    return torch.load(
        path, map_location="cpu", weights_only=True
    )


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def dataset(path, length):
    path = Path(path).resolve()
    extra = {}
    if path.suffix.lower() == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        if manifest.get("format") != "ssm-jepa-corpus-v1":
            raise ValueError("Manifest de corpus incompatible.")
        partitions = []
        for split in ("train", "validation"):
            item = manifest[split]
            source = path.parent / item["file"]
            actual = digest(source)
            if actual != item["sha256"]:
                raise ValueError(f"Corpus modifie : {source}")
            raw = bytearray(source.read_bytes())
            if len(raw) < length + 1:
                raise ValueError(f"Partition trop courte : {split}")
            partitions.append(torch.frombuffer(raw, dtype=torch.uint8).clone())
            extra[split + "_sha256"] = actual
        cut = len(partitions[0])
        data = torch.cat(partitions)
    elif path.suffix.lower() == ".pt":
        data = load(path)
        if not isinstance(data, torch.Tensor) or data.ndim != 1:
            raise ValueError("Le .pt doit contenir un tenseur 1D.")
        if (
            data.is_floating_point()
            or data.min().item() < 0
            or data.max().item() > 255
        ):
            raise ValueError("Octets entiers attendus dans [0,255].")
        data = data.to(torch.uint8)
    else:
        raw = bytearray(path.read_bytes())
        if not raw:
            raise ValueError("Corpus vide.")
        data = torch.frombuffer(raw, dtype=torch.uint8).clone()

    holdout = (
        500_000 if len(data) >= 1_000_000 else len(data) // 5
    )
    if not extra:
        cut = len(data) - holdout
    if min(cut, len(data) - cut) < length + 1:
        raise ValueError(
            f"Texte trop court : au moins "
            f"{5 * (length + 1)} octets conseilles."
        )
    return data, cut, {
        "path": str(path),
        "sha256": digest(path),
        "train_end": cut,
        "bytes": len(data),
        **extra,
    }


def sample(data, count, length, generator):
    starts = torch.randint(
        0, len(data) - length, (count,), generator=generator
    )
    return torch.stack([
        data[i:i + length + 1] for i in starts.tolist()
    ])


def construct(state, device="cpu"):
    if state.get("format") != "ssm-jepa-language-v1":
        raise ValueError("Checkpoint incompatible.")

    model = LanguageModel(**state["architecture"])
    model.load_state_dict(state["model"], strict=True)
    width = state["architecture"]["width"]
    predictor = nn.Sequential(
        nn.Linear(width, width),
        nn.GELU(),
        nn.Linear(width, width),
    )
    predictor.load_state_dict(state["predictor"])
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    teacher.load_state_dict(state["teacher"])
    model.to(device)
    teacher.to(device)
    predictor.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(predictor.parameters()),
        lr=state["lr"],
        weight_decay=0.01,
    )
    optimizer.load_state_dict(state["optimizer"])
    generator = torch.Generator().set_state(state["generator"])
    return model, teacher, predictor, optimizer, generator


def snapshot(state, model, teacher, predictor, optimizer, generator):
    state.update(
        model=model.state_dict(),
        teacher=teacher.state_dict(),
        predictor=predictor.state_dict(),
        optimizer=optimizer.state_dict(),
        generator=generator.get_state(),
    )


def initialize(args):
    if args.run.exists():
        raise ValueError("Checkpoint destination deja present.")

    torch.manual_seed(args.seed)
    architecture = {
        "width": getattr(args, "width", 256),
        "layers": getattr(args, "layers", 6),
        "state": getattr(args, "state_size", 4),
    }
    ffn_mult = getattr(args, "ffn_mult", 0)
    if ffn_mult:
        architecture["ffn_mult"] = ffn_mult
    model = LanguageModel(**architecture)
    if not getattr(args, "from_scratch", False):
        model.load_state_dict(load(args.checkpoint), strict=True)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    width = architecture["width"]
    predictor = nn.Sequential(
        nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    generator = torch.Generator().manual_seed(args.seed)
    length = getattr(args, "context", 256)
    data, cut, metadata = dataset(args.data, length)
    memory = sample(data[:cut], 256, length, generator)

    state = {
        "format": "ssm-jepa-language-v1",
        "architecture": architecture,
        "length": length,
        "block": 32,
        "latent_weight": 0.1,
        "ema": 0.99,
        "lr": args.lr,
        "warmup": getattr(args, "warmup", 0),
        "decay_steps": getattr(args, "decay_steps", 0),
        "variance_weight": getattr(args, "variance_weight", 0.0),
        "step": 0,
        "seed": args.seed,
        "active_data": metadata,
        "initial_checkpoint_sha256": (
            None if getattr(args, "from_scratch", False)
            else digest(args.checkpoint)
        ),
        "memory": memory,
        "protected_memory": 128,
        "memory_seen": 128,
        "history": [],
        "objective": "JEPA-inspired future-block EMA prediction",
    }
    snapshot(
        state, model, teacher, predictor, optimizer, generator
    )
    save(args.run, state)
    print(
        f"Modele initialise : {args.run} | "
        f"{sum(p.numel() for p in model.parameters()):,} parametres"
    )


def train(args):
    state = load(args.run)
    device = getattr(args, "device", "cpu")
    model, teacher, predictor, optimizer, generator = construct(state, device)
    selected = (
        args.data if args.data
        else Path(state["active_data"]["path"])
    )
    data, cut, metadata = dataset(selected, state["length"])
    if args.data is None and metadata != state["active_data"]:
        raise ValueError(
            "Donnees modifiees : utiliser --data explicitement."
        )

    state["active_data"] = metadata
    state["history"].append({
        "start_step": state["step"],
        "data": metadata,
        "requested_steps": args.steps,
    })
    stopped = [False]
    old_handler = signal.signal(
        signal.SIGINT,
        lambda *unused: stopped.__setitem__(0, True),
    )
    model.train()
    predictor.train()
    started = time.perf_counter()

    try:
        for local_step in range(args.steps):
            if stopped[0]:
                break

            incoming = sample(
                data[:cut], 4, state["length"], generator
            )
            indices = torch.randint(
                len(state["memory"]), (2,), generator=generator
            )
            batch = torch.cat([
                incoming, state["memory"][indices]
            ]).long().to(device)
            tokens = batch[:, :-1]
            hidden = model.encode(tokens)
            lm = F.cross_entropy(
                model.head(hidden).reshape(-1, 256),
                batch[:, 1:].reshape(-1),
            )

            block = state["block"]
            width = state["architecture"]["width"]
            with torch.no_grad():
                targets = teacher.encode(tokens).reshape(
                    tokens.size(0), -1, block, width
                ).mean(2)[:, 1:]

            context = hidden[:, block - 1:state["length"] - block:block]
            predicted = predictor(context)
            latent = F.mse_loss(
                F.layer_norm(predicted, (width,)),
                F.layer_norm(targets, (width,)),
            )
            std = torch.sqrt(context.reshape(-1, width).var(0, unbiased=False) + 1e-4)
            variance = F.relu(0.1 - std).mean()
            loss = (lm + state["latent_weight"] * latent
                    + state.get("variance_weight", 0.0) * variance)
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Perte non finie; dernier checkpoint conserve."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = (
                list(model.parameters())
                + list(predictor.parameters())
            )
            nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
            warmup = state.get("warmup", 0)
            warm_factor = min(1.0, (state["step"] + 1)/max(1, warmup))
            decay_steps = state.get("decay_steps", 0)
            progress = min(1.0, max(0.0, (state["step"]-warmup)/max(1, decay_steps-warmup)))
            decay_factor = 0.1 + 0.9*(1+math.cos(math.pi*progress))/2 if decay_steps else 1.0
            for group in optimizer.param_groups:
                group["lr"] = state["lr"] * warm_factor * decay_factor
            optimizer.step()

            with torch.no_grad():
                for target, source in zip(
                    teacher.parameters(), model.parameters()
                ):
                    target.mul_(state["ema"]).add_(
                        source, alpha=1 - state["ema"]
                    )

            # 128 fenetres initiales protegees et 128 en reservoir.
            state["memory_seen"] += 1
            slot = torch.randint(
                state["memory_seen"], (1,), generator=generator
            ).item()
            if slot < 128:
                state["memory"][128 + slot] = incoming[0]

            state["step"] += 1
            if (
                state["step"] % 25 == 0
                or local_step == args.steps - 1
                or stopped[0]
            ):
                print(
                    f"pas={state['step']} | "
                    f"LM={lm.item() / math.log(2):.4f} bpb | "
                    f"latent={latent.item():.4f} | "
                    f"std={std.mean().item():.3f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2g} | "
                    f"{time.perf_counter() - started:.1f} s",
                    flush=True,
                )

            if state["step"] % 100 == 0:
                snapshot(
                    state, model, teacher, predictor,
                    optimizer, generator,
                )
                save(args.run, state)

        snapshot(
            state, model, teacher, predictor, optimizer, generator
        )
        save(args.run, state)
    finally:
        signal.signal(signal.SIGINT, old_handler)

    print(f"Sauvegarde : {args.run} | pas={state['step']}")


@torch.no_grad()
def evaluate(args):
    state = load(args.run)
    model = LanguageModel(**state["architecture"])
    model.load_state_dict(state["model"])
    device = getattr(args, "device", "cpu")
    model.to(device)
    model.eval()
    path = (
        args.data if args.data
        else Path(state["active_data"]["path"])
    )
    length = state["length"]
    data, cut, metadata = dataset(path, length)
    if args.data is None and metadata != state["active_data"]:
        raise ValueError("Donnees modifiees depuis l'entrainement.")

    last = len(data) - length - 1
    starts = [
        cut + (last - cut) * i // max(1, args.samples - 1)
        for i in range(args.samples)
    ]
    total = 0.0
    for offset in range(0, len(starts), 4):
        batch = torch.stack([
            data[i:i + length + 1]
            for i in starts[offset:offset + 4]
        ]).long().to(device)
        total += F.cross_entropy(
            model(batch[:, :-1]).reshape(-1, 256),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        ).item()

    print(json.dumps({
        "step": state["step"],
        "data": metadata,
        "windows": args.samples,
        "bpb": total / (args.samples * length * math.log(2)),
    }, indent=2))


@torch.no_grad()
def generate_text(model, prompt, count, temperature, topk, seed):
    model.eval()
    device = next(model.parameters()).device
    states = None
    for token in prompt.encode("utf-8"):
        logits, states = model.step(torch.tensor([token], device=device), states)

    generator = torch.Generator(device=device).manual_seed(seed)
    generated = []
    for _ in range(count):
        scores = logits[0]
        if temperature == 0:
            token = scores.argmax().item()
        else:
            values, indices = torch.topk(
                scores / temperature, topk
            )
            selected = torch.multinomial(
                values.softmax(-1), 1, generator=generator
            )
            token = indices[selected].item()
        generated.append(token)
        logits, states = model.step(
            torch.tensor([token], device=device), states
        )
    return bytes(generated).decode("utf-8", errors="replace")


def generate(args):
    state = load(args.run)
    model = LanguageModel(**state["architecture"])
    model.load_state_dict(state["model"])
    model.to(getattr(args, "device", "cpu"))
    print(args.prompt + generate_text(
        model, args.prompt, args.bytes,
        args.temperature, args.topk, args.seed,
    ))


def main():
    parser = argparse.ArgumentParser(
        description="SSM-JEPA: train, resume, evaluate, generate"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in ("init", "train", "eval", "generate"):
        child = sub.add_parser(command)
        child.add_argument("--run", type=Path, required=True)

        if command == "init":
            child.add_argument(
                "--checkpoint", type=Path,
                default=Path("lm_SSM-n4_A.pt"),
            )
            child.add_argument(
                "--data", type=Path, default=Path("dataA.pt")
            )
            child.add_argument("--lr", type=float, default=1e-5)
            child.add_argument("--seed", type=int, default=150001)
            child.add_argument("--from-scratch", action="store_true")
            child.add_argument("--width", type=int, default=256)
            child.add_argument("--layers", type=int, default=6)
            child.add_argument("--state-size", type=int, default=4)
            child.add_argument("--ffn-mult", type=int, default=0)
            child.add_argument("--context", type=int, default=256)
            child.add_argument("--warmup", type=int, default=0)
            child.add_argument("--decay-steps", type=int, default=0)
            child.add_argument("--variance-weight", type=float, default=0.0)
        elif command == "train":
            child.add_argument("--data", type=Path)
            child.add_argument("--steps", type=int, default=500)
        elif command == "eval":
            child.add_argument("--data", type=Path)
            child.add_argument("--samples", type=int, default=64)
        else:
            child.add_argument("--prompt", required=True)
            child.add_argument("--bytes", type=int, default=256)
            child.add_argument(
                "--temperature", type=float, default=0.8
            )
            child.add_argument("--topk", type=int, default=40)
            child.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponible.")
    if args.command == "init" and (
        min(args.width, args.layers, args.state_size) < 1 or args.ffn_mult < 0
        or args.context < 64 or args.context % 32 or args.warmup < 0
        or (args.decay_steps and args.decay_steps <= args.warmup)
        or args.decay_steps < 0 or not math.isfinite(args.variance_weight)
        or args.variance_weight < 0
    ):
        parser.error("Configuration du modele ou de l'entrainement invalide.")
    if (
        args.threads < 1
        or getattr(args, "steps", 1) < 1
        or getattr(args, "samples", 1) < 1
    ):
        parser.error("Valeurs positives attendues.")
    if args.command == "init" and (
        not math.isfinite(args.lr) or args.lr <= 0
    ):
        parser.error("Taux d'apprentissage positif attendu.")
    if args.command == "generate" and (
        not args.prompt
        or args.bytes < 0
        or not math.isfinite(args.temperature)
        or args.temperature < 0
        or not 1 <= args.topk <= 256
    ):
        parser.error("Parametres de generation invalides.")

    torch.set_num_threads(args.threads)
    {
        "init": initialize,
        "train": train,
        "eval": evaluate,
        "generate": generate,
    }[args.command](args)


if __name__ == "__main__":
    main()
