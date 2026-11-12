import math, torch, torch.nn as nn, torch.nn.functional as F

def pscan(a, b, T=64):
    # a, b : (B, L, d, n). Retourne h avec h_t = a_t*h_{t-1} + b_t, h_-1 = 0
    B, L, d, n = a.shape
    pad = (-L) % T
    if pad:
        a = torch.cat([a, torch.ones(B, pad, d, n, dtype=a.dtype, device=a.device)], 1)
        b = torch.cat([b, torch.zeros(B, pad, d, n, dtype=b.dtype, device=b.device)], 1)
    nc = a.size(1) // T
    a = a.reshape(B, nc, T, d, n)
    b = b.reshape(B, nc, T, d, n)
    k = 1
    while k < T:
        a_s = torch.cat([torch.ones_like(a[:, :, :k]), a[:, :, :-k]], 2)
        b_s = torch.cat([torch.zeros_like(b[:, :, :k]), b[:, :, :-k]], 2)
        b = b + a * b_s
        a = a * a_s
        k *= 2
    hp = torch.zeros(B, d, n, dtype=a.dtype, device=a.device)
    outs = []
    for c in range(nc):
        hc = b[:, c] + a[:, c] * hp.unsqueeze(1)
        outs.append(hc)
        hp = hc[:, -1]
    h = torch.stack(outs, 1).reshape(B, nc * T, d, n)
    return h[:, :L]

class SelectiveSSM(nn.Module):
    def __init__(s, d, n=16, slow=False, par=True):
        super().__init__()
        s.d, s.n, s.par = d, n, par
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
        if s.par:
            dA = torch.exp(dt.unsqueeze(-1) * A)                       # (B,L,d,n)
            dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)  # (B,L,d,n)
            h = pscan(dA, dBu)
            y = (h * Cm.unsqueeze(2)).sum(-1) + u * s.D
        else:
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
    def __init__(s, din, d, depth=2, slow=False, par=True, n=16):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.blocks = nn.ModuleList([SelectiveSSM(d, n=n, slow=slow, par=par) for _ in range(depth)])
        s.norm = nn.LayerNorm(d)
    def forward(s, x):
        x = s.p(x)
        for b in s.blocks: x = b(x)
        return s.norm(x)
