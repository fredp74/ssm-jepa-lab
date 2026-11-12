import csv
from pathlib import Path

# Reutiliser la preparation, sans executer les entrainements.
source = Path("lm_abc.py").read_text(encoding="utf-8-sig")
marker = "\nbase_state = load("
if source.count(marker) != 1:
    raise RuntimeError("Structure lm_abc.py inattendue")
exec(compile(source.split(marker, 1)[0], "abc_setup", "exec"))

K, WEIGHT = 4, 0.1
OUT = Path("lm_jepa_v1")

def hidden(model, x):
    h = model.emb(x) + model.pos[:x.size(1)]
    h = model.tf(
        h, mask=model.mask[:x.size(1), :x.size(1)], is_causal=True
    )
    return model.norm(h)

@torch.no_grad()
def diagnostic(model, target, predictor, domain):
    data = CTEST if domain == "C" else DATA[domain]
    lo = 0 if domain == "C" else ENDS[domain]
    indices = torch.linspace(
        lo, len(data)-L-2, NEVAL
    ).long().tolist()

    sums = {"jepa": 0.0, "zero": 0.0, "copie": 0.0}
    sum_y = torch.zeros(192, dtype=torch.float64)
    positions = 0

    for start in range(0, NEVAL, 10):
        x = torch.stack([
            data[i:i+L+1] for i in indices[start:start+10]
        ]).long()
        online = hidden(model, x[:, :-1])
        teacher = hidden(target, x[:, :-1])

        y = teacher[:, K:].double()
        pred = predictor(online[:, :-K]).double()
        # Copie du present du meme encodeur cible EMA.
        copied = teacher[:, :-K].double()

        sums["jepa"] += (pred-y).square().sum().item()
        sums["zero"] += y.square().sum().item()
        sums["copie"] += (copied-y).square().sum().item()
        sum_y += y.sum(dim=(0, 1))
        positions += y.shape[0]*y.shape[1]

    elements = positions*192
    scores = {key: value/elements for key, value in sums.items()}
    # Meilleure constante par canal sur cet ensemble d'evaluation.
    variance_sum = sums["zero"] - sum_y.square().sum().item()/positions
    scores["moyenne_oracle"] = max(0.0, variance_sum/elements)
    denom = max(scores["moyenne_oracle"], 1e-12)
    scores["jepa_sur_variance"] = scores["jepa"]/denom
    scores["copie_sur_variance"] = scores["copie"]/denom
    return scores

def norm(grads):
    return math.sqrt(sum(
        g.detach().double().square().sum().item()
        for g in grads if g is not None
    ))

def gradient_diagnostic(model, target, predictor, seed):
    main = torch.Generator().manual_seed(seed)
    ga = torch.Generator().manual_seed(seed+100_000)
    gb = torch.Generator().manual_seed(seed+200_000)
    # Meme construction que le premier batch du run C.
    x = torch.cat([
        batch_c(B, main),
        from_pool(pool_a[:2048], 2, ga),
        from_pool(pool_b[:2048], 2, gb),
    ])

    h = hidden(model, x[:, :-1])
    lm = F.cross_entropy(
        model.head(h).reshape(-1, 256), x[:, 1:].reshape(-1)
    )
    with torch.no_grad():
        ht = hidden(target, x[:, :-1])
    jepa = F.mse_loss(predictor(h[:, :-K]), ht[:, K:])

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
    gp = torch.autograd.grad(jepa, tuple(predictor.parameters()))

    nl, nj = norm(gl), norm(gj)
    dot = sum(
        (a.detach().double()*b.detach().double()).sum().item()
        for a, b in zip(gl, gj) if a is not None and b is not None
    )
    values = {
        "norme_LM": nl,
        "norme_JEPA": nj,
        "rapport_JEPA_pondere_sur_LM": WEIGHT*nj/max(nl, 1e-30),
        "cosinus": dot/max(nl*nj, 1e-30),
        "norme_predicteur": norm(gp),
    }
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError("Diagnostic gradient non fini")
    return values

metric_rows, gradient_rows = [], []
for seed in SEEDS:
    saved = load(OUT / f"Transformer_{seed}_C_jepa.pt")
    if (saved["horizon"], saved["weight"]) != (K, WEIGHT):
        raise ValueError("Configuration du checkpoint inattendue")

    model = new_model(saved["model"]).eval()
    target = new_model(saved["target"]).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    predictor = nn.Sequential(
        nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 192)
    )
    predictor.load_state_dict(saved["predictor"])
    predictor.eval()

    for domain in "ABC":
        scores = diagnostic(model, target, predictor, domain)
        metric_rows.append({"seed": seed, "domain": domain, **scores})
        print(
            f"seed={seed} {domain} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in scores.items()),
            flush=True
        )

    gradients = gradient_diagnostic(model, target, predictor, seed)
    gradient_rows.append({"seed": seed, **gradients})
    print(
        f"GRADIENTS seed={seed} | "
        + " | ".join(f"{k}={v:.4e}" for k, v in gradients.items()),
        flush=True
    )
    del model, target, predictor

for filename, rows in [
    ("diagnostic_latent.csv", metric_rows),
    ("diagnostic_gradients.csv", gradient_rows),
]:
    with (OUT / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

print("Diagnostic termine ; checkpoints inchanges.")
