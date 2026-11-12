import ast
import bisect
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)
L, B, BR = 256, 8, 4
STEPS, LR, WARM = 2000, 1e-3, 100
HOLD, NEVAL = 500_000, 200
SEEDS = (1000, 1001, 1002)
ROOT = Path("lm_abc_v1")
ROOT.mkdir(exist_ok=True)

# Extraire les classes sans executer le programme lm_buffer.py.
source = Path("lm_buffer.py").read_text(encoding="utf-8-sig")
tree = ast.parse(source)
classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "TLM"
]
if len(classes) != 1:
    raise RuntimeError("Definition TLM introuvable ou multiple")
exec(compile(ast.Module(body=classes, type_ignores=[]),
             "lm_buffer_classes", "exec"))

def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)

DATA = {k: load(f"data{k}.pt") for k in "AB"}
CTRAIN = load("dataC_train.pt")
CTEST = load("dataC_test.pt")
manifest = json.loads(Path("dataC_manifest.json").read_text("utf-8"))
ENDS = {k: len(v)-HOLD for k, v in DATA.items()}

# Chaque depart C doit produire 257 octets dans une seule oeuvre.
spans = []
for item in manifest:
    if item["split"] == "train":
        offset, size = int(item["offset"]), int(item["bytes"])
        if offset < 0 or offset + size > len(CTRAIN) or size <= L:
            raise ValueError("Manifeste C incompatible")
        spans.append((offset, size-L))
limits, total = [], 0
for offset, count in spans:
    total += count
    limits.append(total)
if not spans:
    raise ValueError("Aucune oeuvre C d'entrainement")

def batch_ab(k, n, gen):
    starts = torch.randint(0, ENDS[k]-L-1, (n,), generator=gen)
    return torch.stack([
        DATA[k][i:i+L+1] for i in starts.tolist()
    ]).long()

def batch_c(n, gen):
    draws = torch.randint(0, total, (n,), generator=gen).tolist()
    pieces = []
    for draw in draws:
        j = bisect.bisect_right(limits, draw)
        previous = 0 if j == 0 else limits[j-1]
        start = spans[j][0] + draw-previous
        pieces.append(CTRAIN[start:start+L+1])
    return torch.stack(pieces).long()

def make_pool(k, seed):
    slots = ENDS[k] // (L+1)
    if slots < 4096:
        raise ValueError(f"Train {k} trop petit")
    gen = torch.Generator().manual_seed(seed)
    starts = torch.randperm(slots, generator=gen)[:4096]*(L+1)
    return torch.stack([
        DATA[k][i:i+L+1] for i in starts.tolist()
    ]).to(torch.uint8)

pool_a = make_pool("A", 424242)
pool_b = make_pool("B", 424243)

def from_pool(pool, n, gen):
    idx = torch.randint(len(pool), (n,), generator=gen)
    return pool[idx].long()

@torch.no_grad()
def evaluate(model):
    was_training = model.training
    model.eval()
    scores = {}
    for k in "ABC":
        data = CTEST if k == "C" else DATA[k]
        lo = 0 if k == "C" else ENDS[k]
        indices = torch.linspace(
            lo, len(data)-L-2, NEVAL
        ).long().tolist()
        loss_sum = 0.0
        for start in range(0, NEVAL, 50):
            x = torch.stack([
                data[i:i+L+1] for i in indices[start:start+50]
            ]).long()
            loss_sum += F.cross_entropy(
                model(x[:, :-1]).reshape(-1, 256),
                x[:, 1:].reshape(-1), reduction="sum"
            ).item()
        scores[k] = loss_sum/(NEVAL*L*math.log(2))
    model.train(was_training)
    return scores

def new_model(state):
    model = TLM()
    model.load_state_dict(state)
    return model

def train(model, domain, replay, seed, tag):
    model.train()
    # Flux principal identique entre les deux branches C.
    main = torch.Generator().manual_seed(seed)
    ga = torch.Generator().manual_seed(seed+100_000)
    gb = torch.Generator().manual_seed(seed+200_000)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )
    t0 = time.perf_counter()
    for step in range(STEPS):
        lr = LR*min(1.0, (step+1)/WARM)*(
            0.1+0.45*(1+math.cos(math.pi*step/STEPS))
        )
        for group in opt.param_groups:
            group["lr"] = lr

        x = batch_ab("B", B, main) if domain == "B" else batch_c(B, main)
        if replay and domain == "B":
            x = torch.cat([x, from_pool(pool_a, BR, ga)])
        elif replay:
            x = torch.cat([
                x,
                from_pool(pool_a[:2048], 2, ga),
                from_pool(pool_b[:2048], 2, gb),
            ])

        loss = F.cross_entropy(
            model(x[:, :-1]).reshape(-1, 256),
            x[:, 1:].reshape(-1)
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Perte non finie : {tag}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step+1) % 500 == 0:
            minutes = (time.perf_counter()-t0)/60
            print(
                f"{tag} | pas {step+1} | {minutes:.1f} min",
                flush=True
            )

def save_state(model, path):
    temporary = path.with_suffix(".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)

base_state = load("lm_Transformer_A.pt")
model = new_model(base_state)
base = evaluate(model)
del model
print("APRES A :", base, flush=True)
print(
    "C : evaluation sur une oeuvre distincte ; "
    "budget replay C = 2048 fenetres A + 2048 B.",
    flush=True
)

rows = []
for seed in SEEDS:
    bpath = ROOT / f"Transformer_{seed}_B.pt"
    if bpath.exists():
        model = new_model(load(bpath))
        print(f"Reprise checkpoint B, seed={seed}", flush=True)
    else:
        torch.manual_seed(seed)
        model = new_model(base_state)
        train(model, "B", True, seed, f"seed={seed} B+replayA")
        save_state(model, bpath)
    after_b = evaluate(model)
    print(f"APRES B seed={seed} : {after_b}", flush=True)
    del model

    for replay in (False, True):
        cond = "replayAB" if replay else "sans_replay"
        cpath = ROOT / f"Transformer_{seed}_C_{cond}.pt"
        torch.manual_seed(seed)
        if cpath.exists():
            model = new_model(load(cpath))
            print(f"Reprise checkpoint C : {seed}, {cond}", flush=True)
        else:
            model = new_model(load(bpath))
            train(model, "C", replay, seed, f"seed={seed} C {cond}")
            save_state(model, cpath)

        scores = evaluate(model)
        row = {
            "seed": seed, "condition": cond,
            "A_apresA": base["A"],
            **{f"{k}_avantC": after_b[k] for k in "ABC"},
            **{f"{k}_apresC": scores[k] for k in "ABC"},
            "delta_A_C": scores["A"]-after_b["A"],
            "delta_B_C": scores["B"]-after_b["B"],
            "oubli_A_total": scores["A"]-base["A"],
        }
        rows.append(row)
        # Reecrire apres chaque branche pour conserver les scores acquis.
        with (ROOT/"results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"RESULTAT seed={seed} {cond} | "
            f"A {scores['A']:.3f} | B {scores['B']:.3f} | "
            f"C {scores['C']:.3f} | "
            f"deltaA pendant C {row['delta_A_C']:+.3f} | "
            f"deltaB pendant C {row['delta_B_C']:+.3f}",
            flush=True
        )
        del model

print("\nMOYENNE +/- ECART-TYPE entre graines de phase B/C")
for cond in ("sans_replay", "replayAB"):
    selected = [r for r in rows if r["condition"] == cond]
    parts = []
    for key in (
        "A_apresC", "B_apresC", "C_apresC",
        "delta_A_C", "delta_B_C", "oubli_A_total"
    ):
        values = torch.tensor([r[key] for r in selected], dtype=torch.float64)
        parts.append(
            f"{key} {values.mean().item():.3f} +/- {values.std().item():.3f}"
        )
    print(cond+" | "+" | ".join(parts), flush=True)
print(f"Resultats : {ROOT/'results.csv'}", flush=True)
