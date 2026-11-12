import time, torch
from ssm_par import SelectiveSSM

def run(m, x, par):
    m.par = par
    x = x.clone().requires_grad_(True)
    t0 = time.time()
    y = m(x)
    g = torch.autograd.grad(y.pow(2).sum(), [x] + list(m.parameters()))
    return y.detach(), g, time.time() - t0

for slow in (False, True):
    for L in (1, 37, 64, 100, 128, 200):
        torch.manual_seed(0)
        m = SelectiveSSM(64, slow=slow)
        x = torch.randn(4, L, 64)
        y1, g1, _ = run(m, x, False)
        y2, g2, _ = run(m, x, True)
        dy = (y1 - y2).abs().max().item()
        dg = max(((a - b).abs().max() / (a.abs().max() + 1e-8)).item() for a, b in zip(g1, g2))
        print(f"slow={slow!s:5} L={L:4d}  max|dy|={dy:.2e}  max rel dgrad={dg:.2e}")

print("\nVitesse (forward+backward, B=8, d=64, 1 bloc)")
torch.manual_seed(0)
m = SelectiveSSM(64, slow=True)
for L in (128, 512, 1024):
    x = torch.randn(8, L, 64)
    _, _, ts = run(m, x, False)
    _, _, tp = run(m, x, True)
    print(f"L={L:5d}  sequentiel {ts:6.2f}s   parallele {tp:6.2f}s   x{ts/tp:.1f}")
