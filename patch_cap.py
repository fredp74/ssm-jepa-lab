import os
p = os.path.expanduser("~/ssm_jepa/longctx.py")
s = open(p, encoding="utf-8-sig").read()
assert "res = {e:" in s
head = s.split("res = {e:")[0]
tail = '''
STEPS_S = 400
POS = {"fin cue": P - 1, "avant repet": P + G - 1}
_, ftmp = make_batch(4096)
MU, SD = ftmp.mean().item(), ftmp.std().item()

res = {e: [] for e in ENCS}
for ename, make in ENCS.items():
    print("=====", ename, flush=True)
    for seed in SEEDS:
        torch.manual_seed(seed)
        enc = make(C, D)
        heads = nn.ModuleDict({k: nn.Linear(D, 1) for k in POS})
        opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=1e-3)
        te = make_batch(256)
        for st_i in range(STEPS_S):
            x, f = make_batch(B)
            y = ((f - MU) / SD).view(-1, 1)
            z = enc(x)
            losses = {k: F.mse_loss(heads[k](z[:, q]), y) for k, q in POS.items()}
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            if (st_i + 1) % 100 == 0:
                print("   pas", st_i + 1, "/", STEPS_S, " | ".join(k + " " + format(v.item(), ".3f") for k, v in losses.items()), flush=True)
        with torch.no_grad():
            z = enc(te[0])
            yte = ((te[1] - MU) / SD).view(-1, 1)
            r = {k: 1 - F.mse_loss(heads[k](z[:, q]), yte).item() / yte.var().item() for k, q in POS.items()}
        res[ename].append(r)
        print("seed", seed, "R2 test:", " | ".join(k + " " + format(v, ".3f") for k, v in r.items()), flush=True)

print()
print("MOYENNE (ecart-type) du R2 de decodage de f, entrainement supervise")
for ename in ENCS:
    cells = []
    for k in POS:
        a = [x[k] for x in res[ename]]
        cells.append(k + " " + format(st.mean(a), ".3f") + " (" + format(st.stdev(a), ".3f") + ")")
    print(format(ename, "9s") + ": " + " | ".join(cells))
'''
open(os.path.expanduser("~/ssm_jepa/capacity.py"), "w", encoding="utf-8").write(head + tail)
