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

C, D = 4, 64
P = 16
GAPS = [16, 32, 64, 128, 256, 512, 1024]
B, STEPS_PER = 16, 150
SEEDS = [0, 1, 2]
MU, SD = 0.275, 0.45 / math.sqrt(12)

# (constructeur, entraine?)
ENCS = {
    "SSM-lent":       (lambda i, d: SSMEnc(i, d, slow=True, par=True), True),
    "GRU":            (lambda i, d: GRUEnc(i, d), True),
    "SSM-lent figé":  (lambda i, d: SSMEnc(i, d, slow=True, par=True), False),
    "GRU figé":       (lambda i, d: GRUEnc(i, d), False),
}
if len(sys.argv) > 1:
    want = sys.argv[1].split(",")
    ENCS = {k: v for k, v in ENCS.items() if k in want}

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

def evaluate(enc, heads, G):
    xs, fs = make_batch(256, G)
    yte = ((fs - MU) / SD).view(-1, 1)
    zc, zg = [], []
    with torch.no_grad():
        for i in range(0, 256, 32):
            z = enc(xs[i:i + 32])
            zc.append(heads["fin cue"](z[:, P - 1]))
            zg.append(heads["fin gap"](z[:, -1]))
    pc, pg = torch.cat(zc), torch.cat(zg)
    v = yte.var().item()
    return {"fin cue": 1 - F.mse_loss(pc, yte).item() / v,
            "fin gap": 1 - F.mse_loss(pg, yte).item() / v}

res = {e: {G: {"fin cue": [], "fin gap": []} for G in GAPS} for e in ENCS}
print("CURRICULUM LONG avec marqueur, trous:", GAPS, "| threads", torch.get_num_threads(), flush=True)
T0 = time.time()
for ename, (make, trained) in ENCS.items():
    print("=====", ename, "(entraîné)" if trained else "(encodeur figé, têtes seules)", flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        enc = make(C + 1, D)
        heads = nn.ModuleDict({"fin cue": nn.Linear(D, 1), "fin gap": nn.Linear(D, 1)})
        params = list(heads.parameters()) + (list(enc.parameters()) if trained else [])
        opt = torch.optim.AdamW(params, lr=1e-3)
        for G in GAPS:
            tg = time.time()
            for _ in range(STEPS_PER):
                x, f = make_batch(B, G)
                y = ((f - MU) / SD).view(-1, 1)
                if trained:
                    z = enc(x)
                else:
                    with torch.no_grad():
                        z = enc(x)
                loss = F.mse_loss(heads["fin cue"](z[:, P - 1]), y) + F.mse_loss(heads["fin gap"](z[:, -1]), y)
                opt.zero_grad()
                loss.backward()
                if trained:
                    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
            r = evaluate(enc, heads, G)
            for k in r:
                res[ename][G][k].append(r[k])
            print("seed", seed, "trou", G, "R2 test: fin cue", format(r["fin cue"], ".3f"),
                  "| fin gap", format(r["fin gap"], ".3f"),
                  "| %.0fs (total %.1f min)" % (time.time() - tg, (time.time() - T0) / 60), flush=True)

print()
print("MOYENNE (ecart-type) du R2 de decodage de f en fin de trou")
print("trou".ljust(16) + " ".join(format(str(G), ">12") for G in GAPS))
for ename in ENCS:
    cells = []
    for G in GAPS:
        a = res[ename][G]["fin gap"]
        cells.append(format(format(st.mean(a), ".2f") + " (" + format(st.stdev(a), ".2f") + ")", ">12"))
    print(ename.ljust(16) + " ".join(cells))
print()
print("PLAFOND (R2 en fin d'indice, moyenne sur trous et graines)")
for ename in ENCS:
    a = [v for G in GAPS for v in res[ename][G]["fin cue"]]
    print(ename.ljust(16), format(st.mean(a), ".3f"))
