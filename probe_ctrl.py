import math, copy, random, statistics as st, torch, torch.nn as nn, torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(s, d, n=16):
        super().__init__()
        s.d, s.n = d, n
        s.norm = nn.LayerNorm(d)
        s.inp = nn.Linear(d, 2 * d)
        s.dt = nn.Linear(d, d)
        s.B = nn.Linear(d, n)
        s.C = nn.Linear(d, n)
        s.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(d, 1))
        s.D = nn.Parameter(torch.ones(d))
        s.out = nn.Linear(d, d)

    def forward(s, x):
        r = x
        u, g = s.inp(s.norm(x)).chunk(2, -1)
        u = F.silu(u)
        dt = F.softplus(s.dt(u))
        Bm, Cm = s.B(u), s.C(u)
        A = -torch.exp(s.A_log)
        h = torch.zeros(x.size(0), s.d, s.n, device=x.device)
        ys = []
        for t in range(x.size(1)):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dBu = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            h = dA * h + dBu
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + u * s.D
        return r + s.out(y * F.silu(g))

class SSMEnc(nn.Module):
    def __init__(s, din, d, depth=2):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.blocks = nn.ModuleList([SelectiveSSM(d) for _ in range(depth)])
        s.norm = nn.LayerNorm(d)

    def forward(s, x):
        x = s.p(x)
        for b in s.blocks:
            x = b(x)
        return s.norm(x)

class GRUEnc(nn.Module):
    def __init__(s, din, d):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.g = nn.GRU(d, d, num_layers=2, batch_first=True)
        s.norm = nn.LayerNorm(d)

    def forward(s, x):
        return s.norm(s.g(s.p(x))[0])

C, D, K, L, B, W = 4, 64, 4, 32, 32, 8
RA, RB = (0.05, 0.20), (0.30, 0.45)
NA, NB = 300, 300
SEEDS = [0, 1, 2, 3, 4]
ENCS = {"SSM": SSMEnc, "GRU": GRUEnc}
CONDS = [("sans replay", 0), ("replay 16", 16), ("replay 2", 2)]

def make_batch(n, r):
    t = torch.arange(L).float().view(1, L, 1, 1)
    f = torch.rand(n, 1, 1, 3) * (r[1] - r[0]) + r[0]
    ph = torch.rand(n, 1, 1, 3) * 2 * math.pi
    c = torch.arange(C).float().view(1, 1, C, 1) * 0.7
    x = torch.sin(f * t + ph + c).sum(-1)
    return x + 0.1 * torch.randn_like(x)

def build(cls):
    enc = cls(C, D)
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters():
        p.requires_grad = False
    pred = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
    return [enc, tgt, pred, opt]

def train_step(m, x):
    enc, tgt, pred, opt = m
    z = enc(x)
    with torch.no_grad():
        zt = tgt(x)
    jepa = F.mse_loss(pred(z[:, W:-K]), zt[:, W + K:])
    var = F.relu(1.0 - z[:, W:].reshape(-1, D).std(0)).mean()
    opt.zero_grad()
    (jepa + var).backward()
    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        for a, b in zip(tgt.parameters(), enc.parameters()):
            a.mul_(0.99).add_(b.detach(), alpha=0.01)

def run(m, n, sampler):
    for _ in range(n):
        train_step(m, sampler())

@torch.no_grad()
def feats(enc, x):
    z = enc(x)
    Z = z[:, W:-K].reshape(-1, D)
    Y = x[:, W + K:].reshape(-1, C)
    X0 = x[:, W:-K].reshape(-1, C)
    return Z, Y, X0

def fit(Z, Y, lam=1e-2):
    Zb = torch.cat([Z, torch.ones(Z.size(0), 1)], 1)
    A = Zb.T @ Zb + lam * torch.eye(Zb.size(1))
    return torch.linalg.solve(A, Zb.T @ Y)

def score(Wp, Z, Y, X0):
    Zb = torch.cat([Z, torch.ones(Z.size(0), 1)], 1)
    return F.mse_loss(Zb @ Wp, Y).item() / F.mse_loss(X0, Y).item()

def measure(m, sets, Wfix=None):
    enc = m[0]
    fA, fB = feats(enc, sets["pA"]), feats(enc, sets["pB"])
    tA, tB = feats(enc, sets["tA"]), feats(enc, sets["tB"])
    WA, WB = fit(fA[0], fA[1]), fit(fB[0], fB[1])
    out = {"A refit": score(WA, *tA), "B refit": score(WB, *tB)}
    out["A figee"] = score(WA if Wfix is None else Wfix, *tA)
    return out, WA


for ename, cls in ENCS.items():
    A, Bv = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        sets = {"pA": make_batch(256, RA), "pB": make_batch(256, RB),
                "tA": make_batch(128, RA), "tB": make_batch(128, RB)}
        m = build(cls)
        r, _ = measure(m, sets)
        A.append(r["A refit"])
        Bv.append(r["B refit"])
    print(f"{ename} NON ENTRAINE: A refit {st.mean(A):.3f} ({st.stdev(A):.3f}) | B refit {st.mean(Bv):.3f} ({st.stdev(Bv):.3f})")
