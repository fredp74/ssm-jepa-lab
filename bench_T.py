import time, torch, torch.nn as nn, functools
import ssm_par
from ssm_par import SSMEnc

_orig = ssm_par.pscan

class GRUEnc(nn.Module):
    def __init__(s, din, d):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.g = nn.GRU(d, d, num_layers=2, batch_first=True)
        s.norm = nn.LayerNorm(d)
    def forward(s, x):
        return s.norm(s.g(s.p(x))[0])

def bench(m, x, reps=2):
    ts = []
    for i in range(reps + 1):  # le 1er passage = warm-up
        t0 = time.time()
        y = m(x)
        y.pow(2).mean().backward()
        ts.append(time.time() - t0)
    return min(ts[1:])

torch.manual_seed(0)
B = 16
print(f"B={B}, d=64, encodeur complet, forward+backward, torch threads={torch.get_num_threads()}")
for L in (256, 512, 1024):
    x = torch.randn(B, L, 4)
    res = {}
    m = SSMEnc(4, 64, slow=True, par=False)
    res["seq"] = bench(m, x)
    for T in (16, 32, 64, 128):
        ssm_par.pscan = functools.partial(_orig, T=T)
        m = SSMEnc(4, 64, slow=True, par=True)
        res[f"T={T}"] = bench(m, x)
    ssm_par.pscan = _orig
    res["GRU"] = bench(GRUEnc(4, 64), x)
    print(f"L={L:5d}  " + "  ".join(f"{k}:{v:5.2f}s" for k, v in res.items()))
