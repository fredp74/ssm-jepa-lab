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


C, D = 4, 64
P = 16
GAPS = [8, 16, 32, 64, 96]
B, STEPS_PER = 16, 150
SEEDS = [0, 1, 2]
MU, SD = 0.275, 0.45 / math.sqrt(12)
ENCS = {"SSM": lambda i, d: SSMEnc(i, d),
        "SSM-lent": lambda i, d: SSMEnc(i, d, slow=True),
        "GRU": lambda i, d: GRUEnc(i, d)}

def make_batch(n, G):
    L = P + G
    t = torch.arange(L).float().view(1, L, 1)
    c = torch.arange(C).float().view(1, 1, C) * 0.7
    f = torch.rand(n, 1, 1) * 0.45 + 0.05
    ph = torch.rand(n, 1, 1) * 2 * math.pi
    fd = torch.rand(n, 1, 1) * 0.45 + 0.05
    pd = torch.rand(n, 1, 1) * 2 * math.pi
    cue = torch.sin(f * t + ph + c)
    gap = torch.sin(fd * t + pd + c)
    x = torch.where(t.expand(n, L, C) < P, cue, gap)
    x = x + 0.1 * torch.randn_like(x)
    mk = (t.expand(n, L, 1) < P).float()
    return torch.cat([x, mk], -1), f.view(n)

res = {e: {G: {"fin cue": [], "fin gap": []} for G in GAPS} for e in ENCS}
print("CURRICULUM avec marqueur, trous:", GAPS, flush=True)
for ename, make in ENCS.items():
    print("=====", ename, flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        enc = make(C + 1, D)
        heads = nn.ModuleDict({"fin cue": nn.Linear(D, 1), "fin gap": nn.Linear(D, 1)})
        opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=1e-3)
        for G in GAPS:
            te = make_batch(256, G)
            for _ in range(STEPS_PER):
                x, f = make_batch(B, G)
                y = ((f - MU) / SD).view(-1, 1)
                z = enc(x)
                loss = F.mse_loss(heads["fin cue"](z[:, P - 1]), y) + F.mse_loss(heads["fin gap"](z[:, -1]), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                z = enc(te[0])
                yte = ((te[1] - MU) / SD).view(-1, 1)
                r = {"fin cue": 1 - F.mse_loss(heads["fin cue"](z[:, P - 1]), yte).item() / yte.var().item(),
                     "fin gap": 1 - F.mse_loss(heads["fin gap"](z[:, -1]), yte).item() / yte.var().item()}
            for k in r:
                res[ename][G][k].append(r[k])
            print("seed", seed, "trou", G, "R2 test: fin cue", format(r["fin cue"], ".3f"), "| fin gap", format(r["fin gap"], ".3f"), flush=True)

print()
print("MOYENNE (ecart-type) du R2 de decodage de f en fin de trou")
print("trou".ljust(10) + " ".join(format(str(G), ">14") for G in GAPS))
for ename in ENCS:
    cells = []
    for G in GAPS:
        a = res[ename][G]["fin gap"]
        cells.append(format(format(st.mean(a), ".2f") + " (" + format(st.stdev(a), ".2f") + ")", ">14"))
    print(format(ename, "10s") + " ".join(cells))
print()
print("PLAFOND (R2 en fin d'indice, moyenne sur trous et graines)")
for ename in ENCS:
    a = [v for G in GAPS for v in res[ename][G]["fin cue"]]
    print(format(ename, "10s"), format(st.mean(a), ".3f"))
