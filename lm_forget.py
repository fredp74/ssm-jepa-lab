import sys, time, math, functools, torch, torch.nn as nn, torch.nn.functional as F
import ssm_par
from ssm_par import SSMEnc
ssm_par.pscan = functools.partial(ssm_par.pscan, T=16)

L, B, STEPS, LR, WARM = 256, 8, 2000, 1e-3, 100
HOLD = 500_000
NEVAL = 200

class TLM(nn.Module):
    def __init__(s, d, nl):
        super().__init__()
        s.emb = nn.Embedding(256, d)
        s.pos = nn.Parameter(torch.zeros(L, d))
        lay = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        s.tf = nn.TransformerEncoder(lay, nl, enable_nested_tensor=False)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Linear(d, 256)
        s.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(L))
    def forward(s, x):
        h = s.emb(x) + s.pos[:x.size(1)]
        h = s.tf(h, mask=s.mask[:x.size(1), :x.size(1)], is_causal=True)
        return s.head(s.norm(h))

class GLM(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.emb = nn.Embedding(256, d)
        s.g = nn.GRU(d, d, num_layers=2, batch_first=True)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Linear(d, 256)
    def forward(s, x):
        return s.head(s.norm(s.g(s.emb(x))[0]))

class SLM(nn.Module):
    def __init__(s, d, depth, n):
        super().__init__()
        s.emb = nn.Embedding(256, d)
        s.enc = SSMEnc(d, d, depth=depth, slow=True, par=True, n=n)
        s.head = nn.Linear(d, 256)
    def forward(s, x):
        return s.head(s.enc(s.emb(x)))

MODELS = {"Transformer": lambda: TLM(192, 4),
          "GRU": lambda: GLM(384),
          "SSM-n4": lambda: SLM(256, 6, 4)}
if len(sys.argv) > 1:
    MODELS = {k: v for k, v in MODELS.items() if k in sys.argv[1].split(",")}

DATA = {k: torch.load("data%s.pt" % k) for k in "AB"}
TR = {k: (0, len(v) - HOLD) for k, v in DATA.items()}
HO = {k: (len(v) - HOLD, len(v)) for k, v in DATA.items()}

def batch(k, n, lo, hi):
    idx = torch.randint(lo, hi - L - 1, (n,)).tolist()
    return torch.stack([DATA[k][i:i + L + 1] for i in idx]).long()

def entropy0(k):
    c = torch.bincount(DATA[k][:2_000_000].long(), minlength=256).float()
    p = c / c.sum()
    return -(p[p > 0] * torch.log2(p[p > 0])).sum().item()

def bpb(m, k):
    lo, hi = HO[k]
    idx = torch.linspace(lo, hi - L - 2, NEVAL).long().tolist()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, NEVAL, 50):
            x = torch.stack([DATA[k][j:j + L + 1] for j in idx[i:i + 50]]).long()
            tot += F.cross_entropy(m(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1), reduction="sum").item()
    return tot / (NEVAL * L) / math.log(2)

def phase(m, k, tag, T0):
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    lo, hi = TR[k]
    run = 0.0
    for s in range(STEPS):
        lr = LR * min(1.0, (s + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        x = batch(k, B, lo, hi)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        run += loss.item()
        if (s + 1) % 250 == 0:
            print("  %s pas %d | perte train %.3f bpb | %.1f min" % (tag, s + 1, run / 250 / math.log(2), (time.time() - T0) / 60), flush=True)
            run = 0.0

H0 = {k: entropy0(k) for k in "AB"}
print("Entropie d'ordre 0 (plancher a battre) : A %.2f bpb | B %.2f bpb" % (H0["A"], H0["B"]), flush=True)
RES = {}
T0 = time.time()
for name, mk in MODELS.items():
    torch.manual_seed(0)
    m = mk()
    print("=====", name, "| %d parametres" % sum(p.numel() for p in m.parameters()), flush=True)
    r = {"init": (bpb(m, "A"), bpb(m, "B"))}
    phase(m, "A", "A", T0)
    r["apresA"] = (bpb(m, "A"), bpb(m, "B"))
    torch.save(m.state_dict(), "lm_%s_A.pt" % name)
    print("  apres A : A %.3f | B %.3f" % r["apresA"], flush=True)
    phase(m, "B", "B", T0)
    r["apresB"] = (bpb(m, "A"), bpb(m, "B"))
    torch.save(m.state_dict(), "lm_%s_B.pt" % name)
    print("  apres B : A %.3f | B %.3f" % r["apresB"], flush=True)
    RES[name] = r

print()
print("BITS PAR OCTET sur donnees tenues a part (bas = mieux), 1 graine, %d pas par phase" % STEPS)
print("%-12s %14s %14s %14s %10s" % ("", "init A / B", "apres A  A / B", "apres B  A / B", "oubli A"))
for n, r in RES.items():
    print("%-12s %7.2f/%5.2f %8.3f/%5.2f %8.3f/%5.2f %+10.3f" % (n, *r["init"], *r["apresA"], *r["apresB"], r["apresB"][0] - r["apresA"][0]))
