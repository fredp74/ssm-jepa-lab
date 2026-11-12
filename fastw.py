import sys, time, math, statistics as st, torch, torch.nn as nn, torch.nn.functional as F
from ssm_par import SSMEnc

class GRUEnc(nn.Module):
    def __init__(s, din, d):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.g = nn.GRU(d, d, num_layers=2, batch_first=True)
        s.norm = nn.LayerNorm(d)
    def forward(s, x):
        return s.norm(s.g(s.p(x))[0])

C, D, L, SIG = 4, 64, 128, 0.05
B, STEPS = 32, 800
SEEDS = [0, 1, 2]
WIN = {"pre16": (-16, -1), "k0-3": (0, 3), "k4-15": (4, 15), "k16-31": (16, 31), "k32+": (32, 10**9)}

def make(n):
    th = torch.rand(n, 2, 2) * 1.1 + 0.1
    A = torch.zeros(n, 2, C, C)
    for b in range(2):
        c, s = torch.cos(th[:, :, b]), torch.sin(th[:, :, b])
        A[:, :, 2*b, 2*b] = c; A[:, :, 2*b, 2*b+1] = -s
        A[:, :, 2*b+1, 2*b] = s; A[:, :, 2*b+1, 2*b+1] = c
    ts = torch.randint(32, 97, (n,))
    x = torch.zeros(n, L, C)
    x[:, 0] = torch.randn(n, C)
    ar = torch.arange(n)
    for t in range(L - 1):
        At = A[ar, (t + 1 >= ts).long()]
        x[:, t+1] = (At @ x[:, t].unsqueeze(-1)).squeeze(-1) + SIG * torch.randn(n, C)
    return x, ts

class FW(nn.Module):
    def __init__(s, eta0, learn):
        super().__init__()
        s.raw = nn.Parameter(torch.tensor(math.log(eta0 / (1 - eta0))), requires_grad=learn)
    @property
    def eta(s):
        return torch.sigmoid(s.raw)
    def forward(s, x):
        W = torch.zeros(x.size(0), C, C)
        out = []
        for t in range(x.size(1) - 1):
            xt = x[:, t]
            p = (W @ xt.unsqueeze(-1)).squeeze(-1)
            out.append(p)
            err = x[:, t+1] - p
            den = xt.pow(2).sum(-1).view(-1, 1, 1) + 1e-3
            W = W + s.eta * err.unsqueeze(-1) * xt.unsqueeze(1) / den
        return torch.stack(out, 1)

class Pred(nn.Module):
    def __init__(s, enc):
        super().__init__()
        s.enc, s.head = enc, nn.Linear(D, C)
    def forward(s, x):
        return s.head(s.enc(x)[:, :-1])

class Hyb(nn.Module):
    def __init__(s):
        super().__init__()
        s.fw = FW(0.3, True)
        s.enc = GRUEnc(2 * C, D)
        s.head = nn.Linear(D, C)
        nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)
    def forward(s, x):
        P = s.fw(x)
        return P + s.head(s.enc(torch.cat([x[:, :-1], P], -1)))

def evaluate(fn, x, ts):
    with torch.no_grad():
        pr = torch.cat([fn(x[i:i+64]) for i in range(0, len(x), 64)])
    err = ((pr - x[:, 1:]) ** 2).mean(-1) / SIG ** 2
    k = torch.arange(1, L).view(1, -1) - ts.view(-1, 1)
    return {w: err[(k >= a) & (k <= b)].mean().item() for w, (a, b) in WIN.items()}

torch.manual_seed(12345); Xte, tte = make(512)
torch.manual_seed(999); Xva, tva = make(256)
fmt = lambda r: " ".join("%s %7.2f" % (w, r[w]) for w in WIN)

print("ADAPTATION EN LIGNE | erreur / plancher de bruit (1.0 = parfait)", flush=True)
r = evaluate(lambda x: x[:, :-1], Xte, tte)
print("%-12s %s" % ("copie", fmt(r)), flush=True)
best, be = None, 1e9
for eta in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.95):
    rv = evaluate(FW(eta, False), Xva, tva)
    sc = rv["k0-3"] + rv["k4-15"] + rv["k16-31"]
    if sc < be:
        best, be = eta, sc
r = evaluate(FW(best, False), Xte, tte)
print("%-12s %s | eta=%.2f (choisi sur validation)" % ("LMS fixe", fmt(r), best), flush=True)

MODELS = {
    "FW appris": lambda: FW(0.3, True),
    "GRU": lambda: Pred(GRUEnc(C, D)),
    "SSM-lent": lambda: Pred(SSMEnc(C, D, slow=True, par=True)),
    "GRU+FW": lambda: Hyb(),
}
if len(sys.argv) > 1:
    MODELS = {k: v for k, v in MODELS.items() if k in sys.argv[1].split(",")}

T0 = time.time()
RES = {}
for name, mk in MODELS.items():
    RES[name] = []
    npar = None
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = mk()
        npar = sum(p.numel() for p in m.parameters() if p.requires_grad)
        eta_p = [p for n, p in m.named_parameters() if n.endswith("raw")]
        oth = [p for n, p in m.named_parameters() if not n.endswith("raw")]
        groups = ([{"params": oth, "lr": 1e-3}] if oth else []) + [{"params": eta_p, "lr": 2e-2}]
        opt = torch.optim.AdamW(groups)
        for _ in range(STEPS):
            x, _ = make(B)
            loss = F.mse_loss(m(x)[:, 2:], x[:, 3:])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        r = evaluate(m, Xte, tte)
        RES[name].append(r)
        e = (m.eta if isinstance(m, FW) else m.fw.eta).item() if name in ("FW appris", "GRU+FW") else float("nan")
        print("%-10s seed %d %s | eta=%.2f | %.1f min" % (name, seed, fmt(r), e, (time.time() - T0) / 60), flush=True)
    print("  (%s : %d parametres entrainables)" % (name, npar), flush=True)

print()
print("MEDIANE [min-max] sur %d graines" % len(SEEDS))
for name in MODELS:
    cells = []
    for w in WIN:
        a = [r[w] for r in RES[name]]
        cells.append("%s %.2f [%.2f-%.2f]" % (w, st.median(a), min(a), max(a)))
    print("%-10s " % name + " | ".join(cells))
