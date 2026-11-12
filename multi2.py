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

C, D, P, Q, KM = 4, 64, 16, 4, 8
STAGES = [(1, 4), (1, 8), (1, 16), (1, 32), (2, 32), (4, 32), (8, 32)]
STEPS, B = 200, 16
TEST_K, TEST_G = [1, 2, 4, 8], 32
SEEDS = list(range(6))
MU, SD = 0.275, 0.45 / math.sqrt(12)
DIN = C + 2 * KM
ENCS = {"SSM-lent": lambda i, d: SSMEnc(i, d, slow=True, par=True),
        "GRU": lambda i, d: GRUEnc(i, d)}
if len(sys.argv) > 1:
    ENCS = {k: v for k, v in ENCS.items() if k in sys.argv[1].split(",")}

def make_batch(n, K, G):
    L = K * P + G + Q
    tt = torch.arange(L)
    t3 = tt.float().view(1, L, 1)
    c = torch.arange(C).float().view(1, 1, C) * 0.7
    f = torch.rand(n, K) * 0.45 + 0.05
    ph = torch.rand(n, K) * 2 * math.pi
    slot = (tt // P).clamp(max=K - 1)
    cue = torch.sin(f[:, slot].unsqueeze(-1) * t3 + ph[:, slot].unsqueeze(-1) + c)
    fd = torch.rand(n, 1, 1) * 0.45 + 0.05
    pd = torch.rand(n, 1, 1) * 2 * math.pi
    dist = torch.sin(fd * t3 + pd + c)
    incue = (tt < K * P).view(1, L, 1)
    x = torch.where(incue, cue, dist) + 0.1 * torch.randn(n, L, C)
    m = torch.zeros(n, L, KM)
    for k in range(K):
        m[:, k * P:(k + 1) * P, k] = 1.0
    q = torch.randint(0, K, (n,))
    qm = torch.zeros(n, L, KM)
    qm[:, L - Q:, :] = F.one_hot(q, KM).float().unsqueeze(1)
    yq = ((f[torch.arange(n), q] - MU) / SD).view(-1, 1)
    ya = (f - MU) / SD
    return torch.cat([x, m, qm], -1), yq, ya

def evaluate(enc, head, K, n=192, cs=32):
    xs, y, _ = make_batch(n, K, TEST_G)
    with torch.no_grad():
        p = torch.cat([head(enc(xs[i:i + cs])[:, -1]) for i in range(0, n, cs)])
    return 1 - F.mse_loss(p, y).item() / y.var().item(), torch.corrcoef(torch.cat([p, y], 1).T)[0, 1].item()

R = {e: [] for e in ENCS}
print("MULTI-ITEMS curriculum trou+K + tete auxiliaire par slot | stages", STAGES, flush=True)
T0 = time.time()
for ename, make in ENCS.items():
    print("=====", ename, flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        enc = make(DIN, D)
        head, ah = nn.Linear(D, 1), nn.Linear(D, 1)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()) + list(ah.parameters()), lr=1e-3)
        for km, G in STAGES:
            for _ in range(STEPS):
                K = int(torch.randint(1, km + 1, (1,)).item())
                x, yq, ya = make_batch(B, K, G)
                z = enc(x)
                loss = F.mse_loss(head(z[:, -1]), yq)
                idx = [(k + 1) * P - 1 for k in range(K)]
                loss = loss + F.mse_loss(ah(z[:, idx]).squeeze(-1), ya)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
        d = {K: evaluate(enc, head, K) for K in TEST_K}
        R[ename].append(d)
        print("seed", seed, " | ".join("K=%d R2 %.2f corr %.2f" % (K, d[K][0], d[K][1]) for K in TEST_K),
              "| %.1f min" % ((time.time() - T0) / 60), flush=True)

print()
print("RESUME (apprises = R2 a K=1 > 0.5 ; hasard ~ 0)")
for ename in ENCS:
    ok = [r for r in R[ename] if r[1][0] > 0.5]
    print(ename, "apprises: %d/%d" % (len(ok), len(SEEDS)))
    for K in TEST_K:
        if len(ok) > 1:
            print("  K=%d | R2 %.2f (%.2f) | corr %.2f | R2>0.5: %d/%d" % (K, st.mean(r[K][0] for r in ok),
                  st.stdev(r[K][0] for r in ok), st.mean(r[K][1] for r in ok),
                  sum(1 for r in ok if r[K][0] > 0.5), len(ok)))
