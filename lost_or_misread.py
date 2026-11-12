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
TEST_GAPS = [1024, 2048]
B, STEPS_PER = 16, 150
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
    zs = []
    with torch.no_grad():
        for i in range(0, n, 8):
            zs.append(enc(xs[i:i + 8])[:, -1])
        z = torch.cat(zs)
        pred = heads["fin gap"](z)
    r_head = r2(pred, y)
    corr = torch.corrcoef(torch.cat([pred, y], 1).T)[0, 1].item()
    Z = torch.cat([z, torch.ones(n, 1)], 1)
    A = Z[:NF].T @ Z[:NF] + 1.0 * torch.eye(D + 1)
    w = torch.linalg.solve(A, Z[:NF].T @ y[:NF])
    r_refit = r2(Z[NF:] @ w, y[NF:])
    return r_head, corr, pred.mean().item(), pred.std().item(), r_refit

rows = {e: [] for e in ENCS}
print("INFO PERDUE OU MAL LUE | entrainement", TRAIN_GAPS, "| test", TEST_GAPS, flush=True)
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
        torch.save({"enc": enc.state_dict(), "heads": heads.state_dict()},
                   "m_%s_%d.pt" % (ename, seed))
        enc.eval()
        d = {}
        parts = []
        for G in TEST_GAPS:
            rh, co, mp, sp, rr = analyse(enc, heads, G)
            d[G] = (rh, co, mp, sp, rr)
            parts.append("%d: R2tete %.2f corr %.2f moy %.2f std %.2f refit %.2f" % (G, rh, co, mp, sp, rr))
        rows[ename].append(d)
        print("seed", seed, " | ".join(parts), "| %.1f min" % ((time.time() - T0) / 60), flush=True)

print()
print("RESUME par groupe (groupe 'tient' = R2 tete > 0.5 au trou considere ; 'echec' = R2 tete <= 0.5)")
print("graines non apprises (corr a 1024 < 0.3 ET R2 tete < 0.3 aux 2 trous) sont a ignorer a la main")
for ename in ENCS:
    for G in TEST_GAPS:
        for name, cond in (("tient", lambda v: v[0] > 0.5), ("echec", lambda v: v[0] <= 0.5)):
            sel = [r[G] for r in rows[ename] if cond(r[G])]
            if sel:
                print("%-9s trou %4d %-5s n=%d | corr moy %.2f | refit moy %.2f | std pred moy %.2f" % (
                    ename, G, name, len(sel), st.mean(v[1] for v in sel),
                    st.mean(v[4] for v in sel), st.mean(v[3] for v in sel)))
