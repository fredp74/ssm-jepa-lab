import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["SSM-n4"])
parser.add_argument("--seeds", type=int, nargs="+", default=[1000])
parser.add_argument("--check", action="store_true")
args = parser.parse_args()

torch.set_num_threads(8)
L, B, BR = 256, 8, 4
STEPS, LR, WARM = 2000, 1e-3, 100
HOLD, NEVAL = 500_000, 200
SEEDS = tuple(args.seeds)
CONDS = ("sans replay", "tampon4096", "replay complet")

class TLM(nn.Module):
    def __init__(self):
        super().__init__()
        d = 192
        self.emb = nn.Embedding(256, d)
        self.pos = nn.Parameter(torch.zeros(L, d))
        layer = nn.TransformerEncoderLayer(
            d, 4, 4*d, dropout=0.0, batch_first=True,
            norm_first=True, activation="gelu"
        )
        self.tf = nn.TransformerEncoder(
            layer, 4, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 256)
        self.register_buffer(
            "mask", nn.Transformer.generate_square_subsequent_mask(L)
        )

    def forward(self, x):
        h = self.emb(x) + self.pos[:x.size(1)]
        h = self.tf(
            h, mask=self.mask[:x.size(1), :x.size(1)], is_causal=True
        )
        return self.head(self.norm(h))

class GLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, 384)
        self.g = nn.GRU(384, 384, num_layers=2, batch_first=True)
        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, 256)

    def forward(self, x):
        return self.head(self.norm(self.g(self.emb(x))[0]))

DATA = {
    k: torch.load(f"data{k}.pt", map_location="cpu", weights_only=True)
    for k in "AB"
}
for k, data in DATA.items():
    if data.ndim != 1 or len(data) <= HOLD + L + 1:
        raise ValueError(f"Corpus {k} incompatible")

TRAIN_END = {k: len(v) - HOLD for k, v in DATA.items()}

def full_batch(k, count, generator):
    # Meme borne que dans lm_anchor.py pour conserver le protocole.
    starts = torch.randint(
        0, TRAIN_END[k] - L - 1, (count,), generator=generator
    )
    return torch.stack([
        DATA[k][i:i+L+1] for i in starts.tolist()
    ]).long()

# Fenetres disjointes de 257 octets, entierement dans le train A.
slots = TRAIN_END["A"] // (L + 1)
if slots < 4096:
    raise ValueError("Train A trop petit pour le grand tampon")

pool_gen = torch.Generator().manual_seed(424242)
starts = torch.randperm(slots, generator=pool_gen)[:4096] * (L + 1)
pool = torch.stack([
    DATA["A"][i:i+L+1] for i in starts.tolist()
]).to(torch.uint8)
POOLS = {"tampon256": pool[:256], "tampon4096": pool}

def replay_batch(cond, generator):
    if cond == "replay complet":
        return full_batch("A", BR, generator)
    memory = POOLS[cond]
    indices = torch.randint(len(memory), (BR,), generator=generator)
    return memory[indices].long()

@torch.no_grad()
def bpb(model, k):
    was_training = model.training
    model.eval()
    indices = torch.linspace(
        TRAIN_END[k], len(DATA[k]) - L - 2, NEVAL
    ).long().tolist()
    total = 0.0
    for start in range(0, NEVAL, 50):
        x = torch.stack([
            DATA[k][i:i+L+1] for i in indices[start:start+50]
        ]).long()
        total += F.cross_entropy(
            model(x[:, :-1]).reshape(-1, 256),
            x[:, 1:].reshape(-1), reduction="sum"
        ).item()
    model.train(was_training)
    return total / (NEVAL * L * math.log(2))

import functools
import ssm_par
from ssm_par import SSMEnc
ssm_par.pscan = functools.partial(ssm_par.pscan, T=16)

class SLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, 256)
        self.enc = SSMEnc(
            256, 256, depth=6, slow=True, par=True, n=4
        )
        self.head = nn.Linear(256, 256)

    def forward(self, x):
        return self.head(self.enc(self.emb(x)))

factory = SLM
checkpoint = torch.load(
    f"lm_{args.model}_A.pt", map_location="cpu", weights_only=True
)
reference = factory()
reference.load_state_dict(checkpoint)
if args.check:
    torch.manual_seed(123)
    x = torch.randint(0, 256, (1, 37))
    logits = reference(x)
    if logits.shape != (1, 37, 256):
        raise RuntimeError(f"Forme inattendue : {logits.shape}")
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 256),
        x[:, 1:].reshape(-1)
    )
    loss.backward()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Perte non finie")
    for name, parameter in reference.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Gradient absent ou non fini : {name}")
    print(
        "CHECK OK : checkpoint charge, forward et backward finis ; "
        f"{sum(p.numel() for p in reference.parameters())} parametres.",
        flush=True
    )
    raise SystemExit(0)

base_a, base_b = bpb(reference, "A"), bpb(reference, "B")
del reference

# Nom reserve a ce protocole : ne pas reutiliser apres modification.
out = Path(f"lm_buffer_v1_{args.model}.csv")
fields = [
    "model", "seed", "condition", "base_A", "base_B",
    "A", "B", "oubli_A", "minutes"
]
rows = []
if out.exists():
    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError("CSV incompatible")
        rows = list(reader)
done = {(int(r["seed"]), r["condition"]) for r in rows}

print(
    f"{args.model} | checkpoint A={base_a:.3f}, B={base_b:.3f}",
    flush=True
)
print(
    "Tampons fixes : 65792 et 1052672 octets bruts ; "
    "graines de phase B uniquement.",
    flush=True
)

for seed in SEEDS:
    for cond in CONDS:
        if (seed, cond) in done:
            print(f"Deja termine : seed={seed}, {cond}", flush=True)
            continue

        torch.manual_seed(seed)
        model = factory()
        model.load_state_dict(checkpoint)
        model.train()
        gen_b = torch.Generator().manual_seed(seed)
        gen_a = torch.Generator().manual_seed(seed + 100_000)
        opt = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=0.01
        )
        t0 = time.perf_counter()

        for step in range(STEPS):
            lr = LR * min(1.0, (step+1)/WARM) * (
                0.1 + 0.45 * (1 + math.cos(math.pi*step/STEPS))
            )
            for group in opt.param_groups:
                group["lr"] = lr

            xb = full_batch("B", B, gen_b)
            if cond == "sans replay":
                x = xb
            else:
                xa = replay_batch(cond, gen_a)
                x = torch.cat([xb, xa])

            loss = F.cross_entropy(
                model(x[:, :-1]).reshape(-1, 256),
                x[:, 1:].reshape(-1)
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Perte non finie : {seed}, {cond}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if (step+1) % 500 == 0:
                minutes = (time.perf_counter()-t0)/60
                print(
                    f"seed={seed} | {cond:14s} | "
                    f"pas {step+1} | {minutes:.1f} min",
                    flush=True
                )

        a, b = bpb(model, "A"), bpb(model, "B")
        row = dict(zip(fields, [
            args.model, seed, cond, base_a, base_b,
            a, b, a-base_a, (time.perf_counter()-t0)/60
        ]))
        new_file = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        rows.append(row)
        print(
            f"RESULTAT seed={seed} | {cond} | "
            f"A {a:.3f} | B {b:.3f} | oubli {a-base_a:+.3f}",
            flush=True
        )
        del model, opt, loss, x, xb

print("\nMOYENNE +/- ECART-TYPE entre graines de phase B")
for cond in CONDS:
    selected = [r for r in rows if r["condition"] == cond]
    parts = []
    for key in ("A", "B", "oubli_A"):
        values = torch.tensor(
            [float(r[key]) for r in selected], dtype=torch.float64
        )
        sd = values.std().item() if len(values) > 1 else float("nan")
        parts.append(f"{key} {values.mean().item():.3f} +/- {sd:.3f}")
    print(f"{cond:14s} | " + " | ".join(parts), flush=True)
print(f"Resultats sauvegardes : {out}", flush=True)
