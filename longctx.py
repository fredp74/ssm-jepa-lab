import math, statistics as st, copy, torch, torch.nn as nn, torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(s, d, n=16, slow=False):
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
        if slow:
            with torch.no_grad():
                dtv = torch.exp(torch.rand(d) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
                s.dt.bias.copy_(torch.log(torch.expm1(dtv)))
                s.dt.weight.mul_(0.1)

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
    def __init__(s, din, d, depth=2, slow=False):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.blocks = nn.ModuleList([SelectiveSSM(d, slow=slow) for _ in range(depth)])
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

C, D, K, W = 4, 64, 4, 8
P, G, R = 16, 96, 16
L = P + G + R
B, STEPS = 16, 300
SEEDS = [0, 1, 2]
PROBES = {"fin cue": P - 1, "mi-gap": P + G // 2, "avant repet": P + G - 1}
ENCS = {"SSM": lambda i, d: SSMEnc(i, d),
        "SSM-lent": lambda i, d: SSMEnc(i, d, slow=True),
        "GRU": lambda i, d: GRUEnc(i, d)}

def make_batch(n):
    t = torch.arange(L).float().view(1, L, 1)
    c = torch.arange(C).float().view(1, 1, C) * 0.7
    f = torch.rand(n, 1, 1) * 0.45 + 0.05
    ph = torch.rand(n, 1, 1) * 2 * math.pi
    fd = torch.rand(n, 1, 1) * 0.45 + 0.05
    pd = torch.rand(n, 1, 1) * 2 * math.pi
    cue = torch.sin(f * t + ph + c)
    gap = torch.sin(fd * t + pd + c)
    rep = torch.sin(f * (t - (P + G)) + ph + c)
    tt = t.expand(n, L, C)
    x = torch.where(tt < P, cue, torch.where(tt < P + G, gap, rep))
    x = x + 0.1 * torch.randn_like(x)
    return x, f.view(n)

def build(make):
    enc = make(C, D)
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
    return jepa.item()

def ones(Z):
    return torch.cat([Z, torch.ones(Z.size(0), 1)], 1)

def fit(Z, Y, lam=1e-2):
    Zb = ones(Z)
    return torch.linalg.solve(Zb.T @ Zb + lam * torch.eye(Zb.size(1)), Zb.T @ Y)

@torch.no_grad()
def probe(enc, tr, te):
    ztr, zte = enc(tr[0]), enc(te[0])
    mu, sd = tr[1].mean(), tr[1].std()
    ytr = ((tr[1] - mu) / sd).view(-1, 1)
    yte = ((te[1] - mu) / sd).view(-1, 1)
    out = {}
    for name, p in PROBES.items():
        Wp = fit(ztr[:, p], ytr)
        pr = ones(zte[:, p]) @ Wp
        out[name] = 1 - F.mse_loss(pr, yte).item() / yte.var().item()
    return out

res = {e: {"non entraine": [], "entraine": []} for e in ENCS}
for ename, make in ENCS.items():
    print("=====", ename, flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        tr, te = make_batch(512), make_batch(256)
        m = build(make)
        r0 = probe(m[0], tr, te)
        res[ename]["non entraine"].append(r0)
        print(f"seed {seed} non entraine: " + " | ".join(f"{k} {v:.3f}" for k, v in r0.items()), flush=True)
        for s in range(STEPS):
            j = train_step(m, make_batch(B)[0])
            if (s + 1) % 100 == 0:
                print(f"   pas {s + 1}/{STEPS} jepa {j:.4f}", flush=True)
        r1 = probe(m[0], tr, te)
        res[ename]["entraine"].append(r1)
        print(f"seed {seed} entraine    : " + " | ".join(f"{k} {v:.3f}" for k, v in r1.items()), flush=True)

print()
print("MOYENNE (ecart-type) du R2 de decodage de f")
for ename in ENCS:
    print("--", ename)
    for cond, v in res[ename].items():
        cells = []
        for k in PROBES:
            a = [x[k] for x in v]
            cells.append(f"{k} {st.mean(a):.3f} ({st.stdev(a):.3f})")
        print(f"{cond:13s}: " + " | ".join(cells))
