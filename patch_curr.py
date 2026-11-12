import os
p = os.path.expanduser("~/ssm_jepa/longctx.py")
s = open(p, encoding="utf-8-sig").read()
key = "C, D, K, W = 4, 64, 4, 8"
assert key in s
head = s.split(key)[0]
tail = '''
C, D = 4, 64
P = 16
GAPS = [8, 16, 32, 64, 96]
B, STEPS_PER = 16, 150
SEEDS = [0, 1, 2]
MU, SD = 0.275, 0.45 / math.sqrt(12)
ENCS = {"SSM": lambda i, d: SSMEnc(i, d),
        "SSM-lent": lambda i, d: SSMEnc(i, d, slow=True),
        "GRU": lambda i, d: GRUEnc(i, d)}

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

res = {e: {G: {"fin cue": [], "fin gap": []} for G in GAPS} for e in ENCS}
print("CURRICULUM avec marqueur, trous:", GAPS, flush=True)
for ename, make in ENCS.items():
    print("=====", ename, flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        enc = make(C + 1, D)
        heads = nn.ModuleDict({"fin cue": nn.Linear(D, 1), "fin gap": nn.Linear(D, 1)})
        opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=1e-3)
        for G in GAPS:
            te = make_batch(256, G)
            for _ in range(STEPS_PER):
                x, f = make_batch(B, G)
                y = ((f - MU) / SD).view(-1, 1)
                z = enc(x)
                loss = F.mse_loss(heads["fin cue"](z[:, P - 1]), y) + F.mse_loss(heads["fin gap"](z[:, -1]), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                z = enc(te[0])
                yte = ((te[1] - MU) / SD).view(-1, 1)
                r = {"fin cue": 1 - F.mse_loss(heads["fin cue"](z[:, P - 1]), yte).item() / yte.var().item(),
                     "fin gap": 1 - F.mse_loss(heads["fin gap"](z[:, -1]), yte).item() / yte.var().item()}
            for k in r:
                res[ename][G][k].append(r[k])
            print("seed", seed, "trou", G, "R2 test: fin cue", format(r["fin cue"], ".3f"), "| fin gap", format(r["fin gap"], ".3f"), flush=True)

print()
print("MOYENNE (ecart-type) du R2 de decodage de f en fin de trou")
print(format("trou", 10) + " ".join(format(str(G), ">14") for G in GAPS))
for ename in ENCS:
    cells = []
    for G in GAPS:
        a = res[ename][G]["fin gap"]
        cells.append(format(format(st.mean(a), ".2f") + " (" + format(st.stdev(a), ".2f") + ")", ">14"))
    print(format(ename, "10s") + " ".join(cells))
print()
print("PLAFOND (R2 en fin d'indice, moyenne sur trous et graines)")
for ename in ENCS:
    a = [v for G in GAPS for v in res[ename][G]["fin cue"]]
    print(format(ename, "10s"), format(st.mean(a), ".3f"))
'''
open(os.path.expanduser("~/ssm_jepa/curriculum.py"), "w", encoding="utf-8").write(head + tail)
