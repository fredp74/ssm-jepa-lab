import time, functools, torch, torch.nn as nn, torch.nn.functional as F
import ssm_par
from ssm_par import SSMEnc
_orig = ssm_par.pscan

class SLM(nn.Module):
    def __init__(s, d, depth, n):
        super().__init__()
        s.emb = nn.Embedding(256, d)
        s.enc = SSMEnc(d, d, depth=depth, slow=True, par=True, n=n)
        s.head = nn.Linear(d, 256)
    def forward(s, x):
        return s.head(s.enc(s.emb(x)))

B, L, d, depth = 8, 256, 256, 6
print("SSM d=%d x%d, batch %d x %d" % (d, depth, B, L), flush=True)
for n in (16, 8, 4, 2):
    for T in (16, 64):
        ssm_par.pscan = functools.partial(_orig, T=T)
        torch.manual_seed(0)
        m = SLM(d, depth, n)
        npar = sum(p.numel() for p in m.parameters())
        ts = []
        for i in range(4):
            x = torch.randint(0, 256, (B, L + 1))
            t0 = time.time()
            F.cross_entropy(m(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            ts.append(time.time() - t0)
        t = min(ts[1:])
        print("n=%2d T=%2d | %d params | %.2f s/pas | %5.0f octets/s | 10M octets = %.0f min" % (
            n, T, npar, t, B * L / t, 1e7 / (B * L / t) / 60), flush=True)
        del m
