import time, torch, torch.nn as nn, torch.nn.functional as F
from ssm_par import SSMEnc

class TLM(nn.Module):
    def __init__(s, d, nl, L=256):
        super().__init__()
        s.emb = nn.Embedding(256, d)
        s.pos = nn.Parameter(torch.zeros(L, d))
        lay = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        s.tf = nn.TransformerEncoder(lay, nl, enable_nested_tensor=False)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Linear(d, 256)
    def forward(s, x):
        L = x.size(1)
        h = s.emb(x) + s.pos[:L]
        h = s.tf(h, mask=nn.Transformer.generate_square_subsequent_mask(L), is_causal=True)
        return s.head(s.norm(h))

class GLM(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.emb = nn.Embedding(256, d)
        s.g = nn.GRU(d, d, num_layers=2, batch_first=True)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Linear(d, 256)
    def forward(s, x):
        return s.head(s.norm(s.g(s.emb(x))[0]))

class SLM(nn.Module):
    def __init__(s, d, depth):
        super().__init__()
        s.emb = nn.Embedding(256, d)
        s.enc = SSMEnc(d, d, depth=depth, slow=True, par=True)
        s.head = nn.Linear(d, 256)
    def forward(s, x):
        return s.head(s.enc(s.emb(x)))

CONF = [("Transformer d256 x4", lambda: TLM(256, 4)),
        ("GRU d512 x2",         lambda: GLM(512)),
        ("SSM d256 x6",         lambda: SLM(256, 6)),
        ("SSM d256 x12",        lambda: SLM(256, 12))]
B, L = 8, 256
print("torch threads:", torch.get_num_threads(), "| batch %d x %d octets" % (B, L), flush=True)
for name, mk in CONF:
    torch.manual_seed(0)
    m = mk()
    npar = sum(p.numel() for p in m.parameters())
    ts = []
    for i in range(4):
        x = torch.randint(0, 256, (B, L + 1))
        t0 = time.time()
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
        loss.backward()
        ts.append(time.time() - t0)
    t = min(ts[1:])
    tok = B * L
    print("%-20s %9d params | %.2f s/pas | %6.0f octets/s | 10M octets = %.0f min" % (
        name, npar, t, tok / t, 1e7 / (tok / t) / 60), flush=True)
    del m
