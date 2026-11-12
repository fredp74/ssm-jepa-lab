import csv
from pathlib import Path

source = Path("lm_abc.py").read_text(encoding="utf-8-sig")
marker = "\nbase_state = load("
if source.count(marker) != 1:
    raise RuntimeError("Structure lm_abc.py inattendue")
exec(compile(source.split(marker, 1)[0], "abc_setup", "exec"))

ROOT = Path("lm_jepa_block32_v1")
saved = load(ROOT / "Transformer_1000_C_jepa.pt")
if saved.get("objective") != "next_block_mean":
    raise ValueError("Checkpoint incompatible")

K = saved["block_size"]
model = new_model(saved["model"]).eval()
target = new_model(saved["target"]).eval()
predictor = nn.Sequential(
    nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 192)
)
predictor.load_state_dict(saved["predictor"])
predictor.eval()

def hidden(m, x):
    h = m.emb(x) + m.pos[:x.size(1)]
    h = m.tf(
        h, mask=m.mask[:x.size(1), :x.size(1)], is_causal=True
    )
    return m.norm(h)

@torch.no_grad()
def collect(domain):
    data = CTEST if domain == "C" else DATA[domain]
    lo = 0 if domain == "C" else ENDS[domain]
    indices = torch.linspace(
        lo, len(data)-L-2, NEVAL
    ).long().tolist()
    predictions, futures, copies = [], [], []
    for start in range(0, NEVAL, 10):
        x = torch.stack([
            data[i:i+L+1] for i in indices[start:start+10]
        ]).long()
        h = hidden(model, x[:, :-1])
        ht = hidden(target, x[:, :-1])
        blocks = ht.reshape(ht.size(0), L//K, K, 192).mean(2)
        predictions.append(predictor(h[:, K-1:L-K:K]))
        futures.append(blocks[:, 1:])
        copies.append(blocks[:, :-1])
    return [
        torch.cat(parts).double()
        for parts in (predictions, futures, copies)
    ]

rows = []
for domain in "ABC":
    prediction, future, copied = collect(domain)

    def mse(a, b):
        return (a-b).square().mean().item()

    # Constante optimale sur ce domaine de test : controle oracle.
    mean = future.mean(dim=(0, 1), keepdim=True)
    variance = mse(future, mean)
    error = mse(prediction, future)
    copy_error = mse(copied, future)

    # Dix permutations sans point fixe, meme position de bloc.
    gen = torch.Generator().manual_seed(7654321)
    n = future.size(0)
    shuffled_errors = []
    for _ in range(10):
        order = torch.randperm(n, generator=gen)
        permutation = torch.empty(n, dtype=torch.long)
        permutation[order] = order.roll(1)
        shuffled_errors.append(mse(prediction, future[permutation]))
    shuffled_error = sum(shuffled_errors)/len(shuffled_errors)

    # Controle supplementaire : moyenne propre a chaque position de bloc.
    position_mean = future.mean(dim=0, keepdim=True)
    position_error = mse(future, position_mean)
    denom = max(variance, 1e-12)

    row = {
        "domain": domain,
        "jepa": error,
        "variance_domaine": variance,
        "copie": copy_error,
        "melange": shuffled_error,
        "moyenne_par_position_oracle": position_error,
        "jepa_sur_variance": error/denom,
        "copie_sur_variance": copy_error/denom,
        "melange_sur_variance": shuffled_error/denom,
        "gain_vs_melange": 1-error/max(shuffled_error, 1e-12),
    }
    rows.append(row)
    print(
        f"TEST {domain} | "
        + " | ".join(
            f"{key}={value:.4f}"
            for key, value in row.items() if key != "domain"
        ),
        flush=True
    )

out = ROOT / "diagnostic_blocs_test.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Diagnostic termine : {out} ; checkpoints inchanges.")
