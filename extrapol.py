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
TEST_GAPS = [128, 256, 512, 1024, 2048]
B, STEPS_PER = 16, 150
SEEDS = [0, 1, 2, 3, 4]
MU, SD = 0.275, 0.45 / math.sqrt(12)
ENCS = {"SSM-lent": lambda i, d: SSMEnc(i, d, slow=True, par=True),
        "GRU": lambda i, d: GRUEnc(i, d)}
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
    pc, pg = [], []
    with torch.no_grad():
        for i in range(0, n, cs):
            z = enc(xs[i:i + cs])
            pc.append(heads["fin cue"](z[:, P - 1]))
            pg.append(heads["fin gap"](z[:, -1]))
    pc, pg = torch.cat(pc), torch.cat(pg)
    v = y.var().item()
    return 1 - F.mse_loss(pc, y).item() / v, 1 - F.mse_loss(pg, y).item() / v

res = {e: {G: [] for G in TEST_GAPS} for e in ENCS}
cue_res = {e: [] for e in ENCS}
print("EXTRAPOLATION: entrainement trous", TRAIN_GAPS, "| test", TEST_GAPS, flush=True)
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
        out = []
        for G in TEST_GAPS:
            rc, rg = evaluate(enc, heads, G)
            res[ename][G].append(rg)
            cue_res[ename].append(rc)
            out.append("%d:%.3f" % (G, rg))
        print("seed", seed, "R2 fin gap par trou ->", "  ".join(out),
              "| total %.1f min" % ((time.time() - T0) / 60), flush=True)

print()
print("MOYENNE (ecart-type) du R2 en fin de trou, SANS reentrainement (entraine jusqu'a 128)")
print("".ljust(10) + " ".join(format(str(G), ">13") for G in TEST_GAPS))
for ename in ENCS:
    cells = []
    for G in TEST_GAPS:
        a = res[ename][G]
        cells.append(format(format(st.mean(a), ".2f") + " (" + format(st.stdev(a), ".2f") + ")", ">13"))
    print(ename.ljust(10) + " ".join(cells))
print()
print("PLAFOND (R2 en fin d'indice, tous trous et graines)")
for ename in ENCS:
    print(ename.ljust(10), format(st.mean(cue_res[ename]), ".3f"))
