import math, copy, torch, torch.nn as nn, torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(s, d, n=16):
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

class Encoder(nn.Module):
    def __init__(s, din, d, depth=2):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.blocks = nn.ModuleList([SelectiveSSM(d) for _ in range(depth)])
        s.norm = nn.LayerNorm(d)

    def forward(s, x):
        x = s.p(x)
        for b in s.blocks:
            x = b(x)
        return s.norm(x)

def make_batch(B, L, C, dev):
    t = torch.arange(L, device=dev).float().view(1, L, 1, 1)
    f = torch.rand(B, 1, 1, 3, device=dev) * 0.4 + 0.05
    ph = torch.rand(B, 1, 1, 3, device=dev) * 2 * math.pi
    c = torch.arange(C, device=dev).float().view(1, 1, C, 1) * 0.7
    x = torch.sin(f * t + ph + c).sum(-1)
    return x + 0.1 * torch.randn_like(x)

dev = "cuda" if torch.cuda.is_available() else "cpu"
C, D, K, L, B = 4, 64, 4, 64, 64
enc = Encoder(C, D).to(dev)
tgt = copy.deepcopy(enc)
for p in tgt.parameters():
    p.requires_grad = False
pred = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D)).to(dev)
opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
W = 8

for step in range(1, 801):
    x = make_batch(B, L, C, dev)
    z = enc(x)
    with torch.no_grad():
        zt = tgt(x)
    p = pred(z[:, W:-K])
    target = zt[:, W + K:]
    jepa = F.mse_loss(p, target)
    zf = z[:, W:].reshape(-1, D)
    var = F.relu(1.0 - zf.std(0)).mean()
    loss = jepa + 1.0 * var
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        m = 0.99
        for a, b in zip(tgt.parameters(), enc.parameters()):
            a.mul_(m).add_(b.detach(), alpha=1 - m)
    if step % 50 == 0:
        with torch.no_grad():
            base = F.mse_loss(z[:, W:-K].detach(), target)
        print(f"step {step:4d} jepa {jepa.item():.4f} copy_baseline {base.item():.4f} std_z {zf.std(0).mean().item():.3f}")
