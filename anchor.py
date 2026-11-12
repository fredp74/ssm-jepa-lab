import sys, time, math, copy, statistics as st, torch, torch.nn as nn, torch.nn.functional as F
from ssm_par import SSMEnc

class GRUEnc(nn.Module):
    def __init__(s, din, d):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.g = nn.GRU(d, d, num_layers=2, batch_first=True)
        s.norm = nn.LayerNorm(d)
    def forward(s, x):
        return s.norm(s.g(s.p(x))[0])

C, D, L, K, W = 4, 64, 32, 4, 8
STEPS, B, BR, NBUF = 300, 32, 16, 256
SEEDS = list(range(5))
LAM = 1.0
RNG = {"A": (0.05, 0.20), "B": (0.30, 0.45)}
ENCS = {"SSM": lambda: SSMEnc(C, D, slow=False, par=True), "GRU": lambda: GRUEnc(C, D)}
CONDS = ["sans replay", "replay", "replay+ancre", "ancre seule"]
if len(sys.argv) > 1:
    ENCS = {k: v for k, v in ENCS.items() if k in sys.argv[1].split(",")}

def make(n, task):
    lo, hi = RNG[task]
    t = torch.arange(L).float().view(1, L, 1, 1)
    c = torch.arange(C).float().view(1, 1, 1, C) * 0.7
    f = lo + torch.rand(n, 1, 3, 1) * (hi - lo)
    ph = torch.rand(n, 1, 3, 1) * 2 * math.pi
    x = torch.sin(f * t + ph + c).sum(2)
    return x + 0.1 * torch.randn_like(x)

def build(mk):
    enc = mk()
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = nn.Sequential(nn.Linear(D, 128), nn.GELU(), nn.Linear(128, D))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
    return (enc, tgt, pred, opt)

def train(m, task, cond, bufx=None, bufz=None):
    enc, tgt, pred, opt = m
    for _ in range(STEPS):
        xn = make(B, task)
        if cond != "sans replay":
            idx = torch.randint(0, NBUF, (BR,))
            xb, zb = bufx[idx], bufz[idx]
        xin = torch.cat([xn, xb]) if cond in ("replay", "replay+ancre") else xn
        z = enc(xin)
        with torch.no_grad():
            zt = tgt(xin)
        loss = F.mse_loss(pred(z[:, W:-K]), zt[:, W + K:]) + F.relu(1.0 - z[:, W:].reshape(-1, D).std(0)).mean()
        if cond == "replay+ancre":
            loss = loss + LAM * F.mse_loss(z[B:], zb)
        elif cond == "ancre seule":
            loss = loss + LAM * F.mse_loss(enc(xb), zb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for a, b in zip(tgt.parameters(), enc.parameters()):
                a.mul_(0.99).add_(b.detach(), alpha=0.01)

def zs(enc, x):
    with torch.no_grad():
        return torch.cat([enc(x[i:i + 64]) for i in range(0, len(x), 64)])

def lat_ratio(m, x):
    enc, tgt, pred, _ = m
    with torch.no_grad():
        z, zt = zs(enc, x), zs(tgt, x)
        return (F.mse_loss(pred(z[:, W:-K]), zt[:, W + K:]) / F.mse_loss(z[:, W:-K], zt[:, W + K:])).item()

def feats(enc, x):
    z = zs(enc, x)
    Z = torch.cat([z[:, W:L - K].reshape(-1, D), torch.ones(x.size(0) * (L - K - W), 1)], 1)
    return Z, x[:, W + K:L].reshape(-1, C), x[:, W:L - K].reshape(-1, C)

def fit(enc, x):
    Z, Y, _ = feats(enc, x)
    return torch.linalg.solve(Z.T @ Z + torch.eye(D + 1), Z.T @ Y)

def pratio(enc, w, x):
    Z, Y, Xc = feats(enc, x)
    return (F.mse_loss(Z @ w, Y) / F.mse_loss(Xc, Y)).item()

MET = ["latA", "latB", "figee", "refitA", "refitB", "derive"]
res = {e: {c: {k: [] for k in MET} for c in CONDS} for e in ENCS}
T0 = time.time()
print("ANCRAGE LATENT | phase A/B: %d pas | replay %d/batch, tampon %d | lambda %.1f" % (STEPS, BR, NBUF, LAM), flush=True)
for ename, mk in ENCS.items():
    print("=====", ename, flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        m0 = build(mk)
        train(m0, "A", "sans replay")
        enc0 = m0[0]
        teA, teB = make(256, "A"), make(256, "B")
        wA = fit(enc0, make(512, "A"))
        bufx = make(NBUF, "A")
        bufz = zs(enc0, bufx)
        teAz = zs(enc0, teA)
        print("seed %d apres A: latA %.3f | refitA %.3f" % (seed, lat_ratio(m0, teA), pratio(enc0, wA, teA)), flush=True)
        for cond in CONDS:
            m = copy.deepcopy(m0)
            torch.manual_seed(1000 + seed)
            train(m, "B", cond, bufx, bufz)
            enc = m[0]
            r = {"latA": lat_ratio(m, teA), "latB": lat_ratio(m, teB),
                 "figee": pratio(enc, wA, teA),
                 "refitA": pratio(enc, fit(enc, make(512, "A")), teA),
                 "refitB": pratio(enc, fit(enc, make(512, "B")), teB),
                 "derive": F.mse_loss(zs(enc, teA), teAz).item()}
            for k in MET:
                res[ename][c := cond][k].append(r[k])
            print("  %-13s " % cond + " ".join("%s %.3f" % (k, r[k]) for k in MET), flush=True)
        print("  [%.1f min]" % ((time.time() - T0) / 60), flush=True)

print()
print("MEDIANE [min-max] sur %d graines (bas = mieux ; figee > 1 = pire que la copie brute)" % len(SEEDS))
for ename in ENCS:
    print("---", ename)
    print("cond".ljust(14) + " ".join(k.rjust(20) for k in MET))
    for cond in CONDS:
        cells = []
        for k in MET:
            a = res[ename][cond][k]
            cells.append(("%.2f [%.2f-%.2f]" % (st.median(a), min(a), max(a))).rjust(20))
        print(cond.ljust(14) + " ".join(cells))
    print("Diff appariee vs 'replay' (negatif = mieux), graines ameliorees :")
    for cond in ("replay+ancre", "ancre seule"):
        parts = []
        for k in ("figee", "derive", "refitB", "latB"):
            d = [a - b for a, b in zip(res[ename][cond][k], res[ename]["replay"][k])]
            parts.append("%s %+.3f (%d/%d)" % (k, st.mean(d), sum(1 for v in d if v < 0), len(d)))
        print("  %-13s " % cond + " | ".join(parts))
