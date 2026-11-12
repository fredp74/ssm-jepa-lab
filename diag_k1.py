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

C, D, P, Q = 4, 64, 16, 4
STEPS, B = 200, 16
SEEDS = list(range(4))
MU, SD = 0.275, 0.45 / math.sqrt(12)
CONDS = [("direct",     [32, 32, 32, 32], False),
         ("direct+aux", [32, 32, 32, 32], True),
         ("curr",       [4, 8, 16, 32],   False),
         ("curr+aux",   [4, 8, 16, 32],   True)]
ENCS = {"SSM-lent": lambda i, d: SSMEnc(i, d, slow=True, par=True),
        "GRU": lambda i, d: GRUEnc(i, d)}
if len(sys.argv) > 1:
    ENCS = {k: v for k, v in ENCS.items() if k in sys.argv[1].split(",")}

def make_batch(n, G):
    L = P + G + Q
    t = torch.arange(L).float().view(1, L, 1)
    c = torch.arange(C).float().view(1, 1, C) * 0.7
    f = torch.rand(n, 1, 1) * 0.45 + 0.05
    ph = torch.rand(n, 1, 1) * 2 * math.pi
    fd = torch.rand(n, 1, 1) * 0.45 + 0.05
    pd = torch.rand(n, 1, 1) * 2 * math.pi
    cue = torch.sin(f * t + ph + c)
    dist = torch.sin(fd * t + pd + c)
    x = torch.where(t.expand(n, L, C) < P, cue, dist) + 0.1 * torch.randn(n, L, C)
    mk = (t.expand(n, L, 1) < P).float()
    qm = (t.expand(n, L, 1) >= L - Q).float()
    return torch.cat([x, mk, qm], -1), ((f.view(n, 1) - MU) / SD)

def evaluate(enc, head, G, n=192, cs=32):
    xs, y = make_batch(n, G)
    with torch.no_grad():
        p = torch.cat([head(enc(xs[i:i + cs])[:, -1]) for i in range(0, n, cs)])
    return 1 - F.mse_loss(p, y).item() / y.var().item()

R = {(e, c[0]): [] for e in ENCS for c in CONDS}
T0 = time.time()
print("DIAGNOSTIC K=1 | trou final 32 | %d pas par etape" % STEPS, flush=True)
for ename, make in ENCS.items():
    for cname, gaps, aux in CONDS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            enc = make(C + 2, D)
            head, ah = nn.Linear(D, 1), nn.Linear(D, 1)
            opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()) + list(ah.parameters()), lr=1e-3)
            for G in gaps:
                for _ in range(STEPS):
                    x, y = make_batch(B, G)
                    z = enc(x)
                    loss = F.mse_loss(head(z[:, -1]), y)
                    if aux:
                        loss = loss + F.mse_loss(ah(z[:, P - 1]), y)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                    opt.step()
            r = evaluate(enc, head, 32)
            R[(ename, cname)].append(r)
            print("%-9s %-11s seed %d R2 %.3f | %.1f min" % (ename, cname, seed, r, (time.time() - T0) / 60), flush=True)

print()
print("RESUME: R2 a K=1, trou 32 (hasard ~ 0)")
for (e, c), a in R.items():
    print("%-9s %-11s moyenne %.2f | graines R2>0.5: %d/%d" % (e, c, st.mean(a), sum(1 for v in a if v > 0.5), len(a)))
