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
CURR = [16, 32, 64, 128]
B, STEPS_PER, STEPS_VAR = 16, 150, 200
TEST_GAPS = [512, 1024, 2048]
SEEDS = list(range(8))
MU, SD = 0.275, 0.45 / math.sqrt(12)
NF, NT = 256, 128
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

def r2(p, y):
    return 1 - F.mse_loss(p, y).item() / y.var().item()

def analyse(enc, heads, G):
    n = NF + NT
    xs, fs = make_batch(n, G)
    y = ((fs - MU) / SD).view(-1, 1)
    with torch.no_grad():
        z = torch.cat([enc(xs[i:i + 8])[:, -1] for i in range(0, n, 8)])
        pred = heads["fin gap"](z)
    corr = torch.corrcoef(torch.cat([pred, y], 1).T)[0, 1].item()
    Z = torch.cat([z, torch.ones(n, 1)], 1)
    w = torch.linalg.solve(Z[:NF].T @ Z[:NF] + torch.eye(D + 1), Z[:NF].T @ y[:NF])
    return r2(pred, y), corr, r2(Z[NF:] @ w, y[NF:]), pred.mean().item()

def step(enc, heads, opt, G):
    x, f = make_batch(B, G)
    y = ((f - MU) / SD).view(-1, 1)
    z = enc(x)
    loss = F.mse_loss(heads["fin cue"](z[:, P - 1]), y) + F.mse_loss(heads["fin gap"](z[:, -1]), y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
    opt.step()

rows = {e: [] for e in ENCS}
print("TROUS VARIABLES | curriculum", CURR, "+", STEPS_VAR, "pas log-uniforme [16,512] | test", TEST_GAPS, flush=True)
T0 = time.time()
for ename, make in ENCS.items():
    print("=====", ename, flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        enc = make(C + 1, D)
        heads = nn.ModuleDict({"fin cue": nn.Linear(D, 1), "fin gap": nn.Linear(D, 1)})
        opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=1e-3)
        for G in CURR:
            for _ in range(STEPS_PER):
                step(enc, heads, opt, G)
        for _ in range(STEPS_VAR):
            G = int(round(math.exp(random_u := (math.log(16) + torch.rand(1).item() * (math.log(512) - math.log(16))))))
            step(enc, heads, opt, G)
        d, parts = {}, []
        for G in TEST_GAPS:
            d[G] = analyse(enc, heads, G)
            parts.append("%d: tete %.2f corr %.2f refit %.2f moy %.2f" % ((G,) + d[G]))
        rows[ename].append(d)
        print("seed", seed, " | ".join(parts), "| %.1f min" % ((time.time() - T0) / 60), flush=True)

print()
print("RESUME (graines apprises = corr a 512 > 0.5)")
for ename in ENCS:
    ok = [r for r in rows[ename] if r[512][1] > 0.5]
    print(ename, "apprises: %d/%d" % (len(ok), len(SEEDS)))
    for G in TEST_GAPS:
        if len(ok) > 1:
            print("  trou %4d | R2 tete %.2f (%.2f) | refit %.2f (%.2f) | tete>0.5: %d/%d" % (
                G, st.mean(r[G][0] for r in ok), st.stdev(r[G][0] for r in ok),
                st.mean(r[G][2] for r in ok), st.stdev(r[G][2] for r in ok),
                sum(1 for r in ok if r[G][0] > 0.5), len(ok)))
