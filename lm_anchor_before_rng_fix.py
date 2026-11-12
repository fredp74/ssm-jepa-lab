import sys, time, math, copy, torch, torch.nn as nn, torch.nn.functional as F

L, B, BR, STEPS, LR, WARM, LAM = 256, 8, 4, 2000, 1e-3, 100, 1.0
HOLD, NEVAL = 500_000, 200

class TLM(nn.Module):
    def __init__(s, d, nl):
        super().__init__()
        s.emb = nn.Embedding(256, d)
        s.pos = nn.Parameter(torch.zeros(L, d))
        lay = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        s.tf = nn.TransformerEncoder(lay, nl, enable_nested_tensor=False)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Linear(d, 256)
        s.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(L))
    def hid(s, x):
        h = s.emb(x) + s.pos[:x.size(1)]
        return s.norm(s.tf(h, mask=s.mask[:x.size(1), :x.size(1)], is_causal=True))
    def forward(s, x):
        return s.head(s.hid(x))

class GLM(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.emb = nn.Embedding(256, d)
        s.g = nn.GRU(d, d, num_layers=2, batch_first=True)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Linear(d, 256)
    def hid(s, x):
        return s.norm(s.g(s.emb(x))[0])
    def forward(s, x):
        return s.head(s.hid(x))

MODELS = {"Transformer": lambda: TLM(192, 4), "GRU": lambda: GLM(384)}
if len(sys.argv) > 1:
    MODELS = {k: v for k, v in MODELS.items() if k in sys.argv[1].split(",")}
CONDS = ["sans replay", "replay", "replay+ancre", "ancre seule"]

DATA = {k: torch.load("data%s.pt" % k) for k in "AB"}
TR = {k: (0, len(v) - HOLD) for k, v in DATA.items()}
HO = {k: (len(v) - HOLD, len(v)) for k, v in DATA.items()}

def batch(k, n):
    lo, hi = TR[k]
    idx = torch.randint(lo, hi - L - 1, (n,)).tolist()
    return torch.stack([DATA[k][i:i + L + 1] for i in idx]).long()

def bpb(m, k):
    lo, hi = HO[k]
    idx = torch.linspace(lo, hi - L - 2, NEVAL).long().tolist()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, NEVAL, 50):
            x = torch.stack([DATA[k][j:j + L + 1] for j in idx[i:i + 50]]).long()
            tot += F.cross_entropy(m(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1), reduction="sum").item()
    return tot / (NEVAL * L) / math.log(2)

RES, T0 = {}, time.time()
for name, mk in MODELS.items():
    m0 = mk()
    m0.load_state_dict(torch.load("lm_%s_A.pt" % name))
    frozen = copy.deepcopy(m0).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    base = (bpb(m0, "A"), bpb(m0, "B"))
    print("=====", name, "| apres A (checkpoint): A %.3f | B %.3f" % base, flush=True)
    for cond in CONDS:
        m = copy.deepcopy(m0)
        torch.manual_seed(1000)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
        for s in range(STEPS):
            lr = LR * min(1.0, (s + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
            for g in opt.param_groups:
                g["lr"] = lr
            xb = batch("B", B)
            if cond == "sans replay":
                x = xb
            else:
                xa = batch("A", BR)
                x = torch.cat([xb, xa]) if cond != "ancre seule" else xb
            h = m.hid(x[:, :-1])
            lg = m.head(h)
            if cond == "ancre seule":
                loss = F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1))
            else:
                loss = F.cross_entropy(lg.reshape(-1, 256), x[:, 1:].reshape(-1))
            if cond in ("replay+ancre", "ancre seule"):
                xa2 = x[B:] if cond == "replay+ancre" else xa
                ha = h[B:] if cond == "replay+ancre" else m.hid(xa2[:, :-1])
                with torch.no_grad():
                    hf = frozen.hid(xa2[:, :-1])
                loss = loss + LAM * F.mse_loss(ha, hf)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if (s + 1) % 500 == 0:
                print("  %-13s pas %d | %.1f min" % (cond, s + 1, (time.time() - T0) / 60), flush=True)
        r = (bpb(m, "A"), bpb(m, "B"))
        RES[(name, cond)] = r
        print("  %-13s apres B : A %.3f | B %.3f | oubli A %+.3f" % (cond, r[0], r[1], r[0] - base[0]), flush=True)

print()
print("BITS PAR OCTET (bas = mieux), 1 graine ; oubli = bpb A apres B moins bpb A apres A")
for (n, c), r in RES.items():
    print("%-12s %-13s A %.3f | B %.3f" % (n, c, r[0], r[1]))
