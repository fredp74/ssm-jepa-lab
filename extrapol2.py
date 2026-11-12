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

C, D, P = 4, 64, 16
TRAIN_GAPS = [16, 32, 64, 128]
TEST_GAPS = [128, 1024, 2048]
B, STEPS_PER = 16, 150
SEEDS = list(range(8))
MU, SD = 0.275, 0.45 / math.sqrt(12)

def set_dt(enc, dtmin, dtmax=0.1):
    with torch.no_grad():
        for b in enc.blocks:
            dtv = torch.exp(torch.rand(b.d) * (math.log(dtmax) - math.log(dtmin)) + math.log(dtmin))
            b.dt.bias.copy_(torch.log(torch.expm1(dtv)))
    return enc

ENCS = {
    "SSM-lent": lambda i, d: SSMEnc(i, d, slow=True, par=True),
    "SSM-1e-4": lambda i, d: set_dt(SSMEnc(i, d, slow=True, par=True), 1e-4),
    "GRU":      lambda i, d: GRUEnc(i, d),
}
if len(sys.argv) > 1:
    ENCS = {k: v for k, v in ENCS.items() if k in sys.argv[1].split(",")}

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

def evaluate(enc, heads, G, n=128, cs=8):
    xs, fs = make_batch(n, G)
    y = ((fs - MU) / SD).view(-1, 1)
    pg = []
    with torch.no_grad():
        for i in range(0, n, cs):
            pg.append(heads["fin gap"](enc(xs[i:i + cs])[:, -1]))
    pg = torch.cat(pg)
    return 1 - F.mse_loss(pg, y).item() / y.var().item(), pg.std().item()

def slow_frac(enc, thr=2048):
    x = make_batch(8, 512)[0]
    out = []
    with torch.no_grad():
        h = enc.p(x)
        for b in enc.blocks:
            u, _ = b.inp(b.norm(h)).chunk(2, -1)
            u = F.silu(u)
            dt = F.softplus(b.dt(u)).mean((0, 1))
            tau = 1.0 / (dt.unsqueeze(-1) * torch.exp(b.A_log))
            out.append((tau > thr).float().mean().item())
            h = b(h)
    return out

R = {e: {G: [] for G in TEST_GAPS} for e in ENCS}
print("EXTRAPOLATION v2 | entrainement", TRAIN_GAPS, "| test", TEST_GAPS, flush=True)
T0 = time.time()
for ename, make in ENCS.items():
    print("=====", ename, flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        enc = make(C + 1, D)
        heads = nn.ModuleDict({"fin cue": nn.Linear(D, 1), "fin gap": nn.Linear(D, 1)})
        opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=1e-3)
        for G in TRAIN_GAPS:
            for _ in range(STEPS_PER):
                x, f = make_batch(B, G)
                y = ((f - MU) / SD).view(-1, 1)
                z = enc(x)
                loss = F.mse_loss(heads["fin cue"](z[:, P - 1]), y) + F.mse_loss(heads["fin gap"](z[:, -1]), y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
        cells = []
        for G in TEST_GAPS:
            r, ps = evaluate(enc, heads, G)
            R[ename][G].append(r)
            cells.append("%d: R2 %.3f (std pred %.2f)" % (G, r, ps))
        sf = slow_frac(enc) if ename != "GRU" else None
        print("seed", seed, " | ".join(cells),
              ("| tau>2048 bloc1/bloc2: %.2f/%.2f" % tuple(sf)) if sf else "",
              "| %.1f min" % ((time.time() - T0) / 60), flush=True)

print()
print("RESUME: R2 fin de trou, sans reentrainement. 'appris' = R2(128) > 0.5")
for ename in ENCS:
    ok = [i for i, v in enumerate(R[ename][128]) if v > 0.5]
    print(ename.ljust(10), "graines apprises: %d/%d" % (len(ok), len(SEEDS)))
    for G in TEST_GAPS:
        a = [R[ename][G][i] for i in ok]
        if len(a) > 1:
            n_ok = sum(1 for v in a if v > 0.5)
            print("   trou %5d : moyenne %.2f (ecart-type %.2f) | graines R2>0.5: %d/%d" % (G, st.mean(a), st.stdev(a), n_ok, len(a)))
