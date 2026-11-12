import argparse
import copy
import csv
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

# Reutiliser la preparation et les fonctions, sans lancer les anciens runs.
source = Path("lm_abc.py").read_text(encoding="utf-8-sig")
marker = "\nbase_state = load("
if source.count(marker) != 1:
    raise RuntimeError("Structure de lm_abc.py inattendue")
prefix = source.split(marker, 1)[0]
old_root = 'ROOT = Path("lm_abc_v1")'
if prefix.count(old_root) != 1:
    raise RuntimeError("Repertoire de reference introuvable")
prefix = prefix.replace(old_root, 'ROOT = Path("lm_jepa_lam1_v1")', 1)
exec(compile(prefix, "lm_abc_setup", "exec"))

REF = Path("lm_abc_v1")
K, WEIGHT, EMA = 4, 1.0, 0.99
SEEDS = (1000,)

def hidden(model, x):
    h = model.emb(x) + model.pos[:x.size(1)]
    h = model.tf(
        h, mask=model.mask[:x.size(1), :x.size(1)], is_causal=True
    )
    return model.norm(h)

def experiment(seed, smoke=False):
    bpath = REF / f"Transformer_{seed}_B.pt"
    cpath = REF / f"Transformer_{seed}_C_replayAB.pt"
    if not bpath.exists() or not cpath.exists():
        raise FileNotFoundError(f"Checkpoints de reference manquants : {seed}")

    torch.manual_seed(seed)
    model = new_model(load(bpath))
    target = copy.deepcopy(model).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    predictor = nn.Sequential(
        nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 192)
    )
    parameters = list(model.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(parameters, lr=LR, weight_decay=0.01)

    main = torch.Generator().manual_seed(seed)
    ga = torch.Generator().manual_seed(seed+100_000)
    gb = torch.Generator().manual_seed(seed+200_000)
    steps = 2 if smoke else STEPS
    start = time.perf_counter()
    totals = [0.0, 0.0]
    model.train()
    predictor.train()

    for step in range(steps):
        lr = LR*min(1.0, (step+1)/WARM)*(
            0.1+0.45*(1+math.cos(math.pi*step/STEPS))
        )
        for group in opt.param_groups:
            group["lr"] = lr

        x = torch.cat([
            batch_c(B, main),
            from_pool(pool_a[:2048], 2, ga),
            from_pool(pool_b[:2048], 2, gb),
        ])
        h = hidden(model, x[:, :-1])
        lm = F.cross_entropy(
            model.head(h).reshape(-1, 256),
            x[:, 1:].reshape(-1)
        )
        with torch.no_grad():
            ht = hidden(target, x[:, :-1])

        prediction = predictor(h[:, :-K])
        jepa = F.mse_loss(prediction, ht[:, K:])
        loss = lm + WEIGHT*jepa
        if not torch.isfinite(loss).item():
            raise RuntimeError("Perte non finie")

        if (step+1) % 500 == 0:
            encoder = [
                p for name, p in model.named_parameters()
                if not name.startswith("head.")
            ]
            gl = torch.autograd.grad(
                lm, encoder, retain_graph=True, allow_unused=True
            )
            gj = torch.autograd.grad(
                jepa, encoder, retain_graph=True, allow_unused=True
            )
            def gradient_norm(grads):
                return math.sqrt(sum(
                    g.detach().double().square().sum().item()
                    for g in grads if g is not None
                ))
            nl, nj = gradient_norm(gl), gradient_norm(gj)
            dot = sum(
                (a.detach().double()*b.detach().double()).sum().item()
                for a, b in zip(gl, gj)
                if a is not None and b is not None
            )
            print(
                f"GRADIENTS seed={seed} pas={step+1} | "
                f"JEPA_pondere_sur_LM={WEIGHT*nj/max(nl, 1e-30):.4f} | "
                f"cosinus={dot/max(nl*nj, 1e-30):.4f}",
                flush=True
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if smoke:
            if any(p.grad is not None for p in target.parameters()):
                raise RuntimeError("Gradient inattendu dans la cible EMA")
            grad = predictor[-1].weight.grad
            if grad is None or not torch.isfinite(grad).all() or grad.abs().max() == 0:
                raise RuntimeError("Gradient du predicteur invalide")
        nn.utils.clip_grad_norm_(parameters, 1.0)
        opt.step()

        with torch.no_grad():
            for pt, po in zip(target.parameters(), model.parameters()):
                pt.mul_(EMA).add_(po, alpha=1-EMA)

        totals[0] += lm.item()
        totals[1] += jepa.item()
        if smoke or (step+1) % 500 == 0:
            count = step+1 if smoke else 500
            # Dispersion entre positions/sequences, moyenne sur les canaux.
            online_std = h.detach().reshape(-1, 192).std(
                dim=0, unbiased=False
            ).mean().item()
            target_std = ht.reshape(-1, 192).std(
                dim=0, unbiased=False
            ).mean().item()
            print(
                f"seed={seed} pas={step+1} | "
                f"LM {totals[0]/count/math.log(2):.3f} bpb | "
                f"JEPA {totals[1]/count:.4f} | "
                f"std online/cible {online_std:.3f}/{target_std:.3f} | "
                f"{(time.perf_counter()-start)/60:.1f} min",
                flush=True
            )
            if not smoke:
                totals = [0.0, 0.0]

    minutes = (time.perf_counter()-start)/60
    if smoke:
        print("SMOKE OK : pertes finies, gradient predicteur, cible sans gradient.")
        return

    # Sauvegarder aussi cible et predicteur pour conserver l'experience.
    path = ROOT / f"Transformer_{seed}_C_jepa.pt"
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "target": target.state_dict(),
        "predictor": predictor.state_dict(),
        "seed": seed, "horizon": K,
        "weight": WEIGHT, "ema": EMA,
        "training_minutes": minutes,
    }, temporary)
    temporary.replace(path)

if args.smoke:
    experiment(1000, smoke=True)
else:
    rows = []
    for seed in SEEDS:
        path = ROOT / f"Transformer_{seed}_C_jepa.pt"
        if not path.exists():
            experiment(seed)
        saved = load(path)
        before = new_model(load(REF / f"Transformer_{seed}_B.pt"))
        before_scores = evaluate(before)
        del before

        for condition in ("replayAB", "replayAB_JEPA"):
            if condition == "replayAB":
                state = load(REF / f"Transformer_{seed}_C_replayAB.pt")
            else:
                state = saved["model"]
            model = new_model(state)
            scores = evaluate(model)
            del model
            row = {
                "seed": seed, "condition": condition,
                **{f"{k}_avantC": before_scores[k] for k in "ABC"},
                **{f"{k}_apresC": scores[k] for k in "ABC"},
                "delta_A_C": scores["A"]-before_scores["A"],
                "delta_B_C": scores["B"]-before_scores["B"],
                "training_minutes": (
                    saved["training_minutes"]
                    if condition == "replayAB_JEPA" else ""
                ),
            }
            rows.append(row)
            print(
                f"RESULTAT seed={seed} {condition} | "
                f"A {scores['A']:.3f} | B {scores['B']:.3f} | "
                f"C {scores['C']:.3f}", flush=True
            )

        with (ROOT / "results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\nMOYENNE +/- ECART-TYPE")
    for condition in ("replayAB", "replayAB_JEPA"):
        selected = [r for r in rows if r["condition"] == condition]
        parts = []
        for key in ("A_apresC", "B_apresC", "C_apresC"):
            v = torch.tensor([r[key] for r in selected], dtype=torch.float64)
            parts.append(f"{key} {v.mean():.3f} +/- {v.std():.3f}")
        print(condition+" | "+" | ".join(parts), flush=True)
    print(f"Resultats : {ROOT / 'results.csv'}")
