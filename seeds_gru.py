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

class Encoder(nn.Module):
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

C, D, K, L, B, W = 4, 64, 4, 32, 32, 8
RA, RB = (0.05, 0.20), (0.30, 0.45)
NA, NB = 300, 300
SEEDS = [0, 1, 2, 3, 4]
CONDS = [("sans replay", 0), ("replay 16", 16), ("replay 2", 2)]

class GRUEnc(nn.Module):
    def __init__(s, din, d):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.g = nn.GRU(d, d, num_layers=2, batch_first=True)
        s.norm = nn.LayerNorm(d)

    def forward(s, x):
        return s.norm(s.g(s.p(x))[0])

def make_batch(n, r):
    t = torch.arange(L).float().view(1, L, 1, 1)
    f = torch.rand(n, 1, 1, 3) * (r[1] - r[0]) + r[0]
    ph = torch.rand(n, 1, 1, 3) * 2 * math.pi
    c = torch.arange(C).float().view(1, 1, C, 1) * 0.7
    x = torch.sin(f * t + ph + c).sum(-1)
    return x + 0.1 * torch.randn_like(x)

def build():
    enc = GRUEnc(C, D)
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

@torch.no_grad()
def evaluate(m, x):
    enc, tgt, pred, _ = m
    z = enc(x)
    zt = tgt(x)
    j = F.mse_loss(pred(z[:, W:-K]), zt[:, W + K:]).item()
    b = F.mse_loss(z[:, W:-K], zt[:, W + K:]).item()
    return j / b

def run(m, n, sampler):
    for _ in range(n):
        train_step(m, sampler())

print("params SSM", sum(q.numel() for q in Encoder(C, D).parameters()), "| GRU", sum(q.numel() for q in GRUEnc(C, D).parameters()))
res = {"apres A": []}
for name, _ in CONDS:
    res[name] = []

for seed in SEEDS:
    torch.manual_seed(seed)
    random.seed(seed)
    evA, evB = make_batch(128, RA), make_batch(128, RB)
    pool = [make_batch(B, RA) for _ in range(16)]
    m = build()
    run(m, NA, lambda: make_batch(B, RA))
    ra = (evaluate(m, evA), evaluate(m, evB))
    res["apres A"].append(ra)
    print(f"seed {seed} apres A      : A {ra[0]:.3f} | B {ra[1]:.3f}", flush=True)
    ck = [copy.deepcopy(o.state_dict()) for o in m]
    for name, nbuf in CONDS:
        torch.manual_seed(seed + 100)
        m2 = build()
        for o, s in zip(m2, ck):
            o.load_state_dict(s)
        buf = pool[:nbuf]
        if nbuf == 0:
            sampler = lambda: make_batch(B, RB)
        else:
            sampler = lambda buf=buf: torch.cat([make_batch(B // 2, RB), random.choice(buf)[: B // 2]])
        run(m2, NB, sampler)
        r = (evaluate(m2, evA), evaluate(m2, evB))
        res[name].append(r)
        print(f"seed {seed} {name:12s}: A {r[0]:.3f} | B {r[1]:.3f}", flush=True)

print()
print("MOYENNE sur", len(SEEDS), "graines (ecart-type entre parentheses)")
for k, v in res.items():
    a = [x[0] for x in v]
    b = [x[1] for x in v]
    print(f"{k:12s}: A {st.mean(a):.3f} ({st.stdev(a):.3f}) | B {st.mean(b):.3f} ({st.stdev(b):.3f})")
