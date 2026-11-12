import os
p = os.path.expanduser("~/ssm_jepa/probe.py")
s = open(p, encoding="utf-8-sig").read()
head = s.split("res = {e:")[0]
tail = '''
for ename, cls in ENCS.items():
    A, Bv = [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        sets = {"pA": make_batch(256, RA), "pB": make_batch(256, RB),
                "tA": make_batch(128, RA), "tB": make_batch(128, RB)}
        m = build(cls)
        r, _ = measure(m, sets)
        A.append(r["A refit"])
        Bv.append(r["B refit"])
    print(f"{ename} NON ENTRAINE: A refit {st.mean(A):.3f} ({st.stdev(A):.3f}) | B refit {st.mean(Bv):.3f} ({st.stdev(Bv):.3f})")
'''
open(os.path.expanduser("~/ssm_jepa/probe_ctrl.py"), "w", encoding="utf-8").write(head + tail)
