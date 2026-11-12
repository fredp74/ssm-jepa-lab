from pathlib import Path

src = Path("lm_jepa.py").read_text(encoding="utf-8-sig")

patches = [
    (
        'ROOT = Path("lm_jepa_v1")',
        'ROOT = Path("lm_jepa_lam1_v1")'
    ),
    (
        'K, WEIGHT, EMA = 4, 0.1, 0.99',
        'K, WEIGHT, EMA = 4, 1.0, 0.99\nSEEDS = (1000,)'
    ),
    (
        '        opt.zero_grad(set_to_none=True)\n        loss.backward()',
        '''        if (step+1) % 500 == 0:
            encoder = [
                p for name, p in model.named_parameters()
                if not name.startswith("head.")
            ]
            gl = torch.autograd.grad(
                lm, encoder, retain_graph=True, allow_unused=True
            )
            gj = torch.autograd.grad(
                jepa, encoder, retain_graph=True, allow_unused=True
            )
            def gradient_norm(grads):
                return math.sqrt(sum(
                    g.detach().double().square().sum().item()
                    for g in grads if g is not None
                ))
            nl, nj = gradient_norm(gl), gradient_norm(gj)
            dot = sum(
                (a.detach().double()*b.detach().double()).sum().item()
                for a, b in zip(gl, gj)
                if a is not None and b is not None
            )
            print(
                f"GRADIENTS seed={seed} pas={step+1} | "
                f"JEPA_pondere_sur_LM={WEIGHT*nj/max(nl, 1e-30):.4f} | "
                f"cosinus={dot/max(nl*nj, 1e-30):.4f}",
                flush=True
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()'''
    ),
]

for old, new in patches:
    if src.count(old) != 1:
        raise SystemExit(
            f"Creation interrompue : motif absent ou multiple : {old!r}"
        )
    src = src.replace(old, new, 1)

compile(src, "lm_jepa_lam1.py", "exec")
Path("lm_jepa_lam1.py").write_text(src, encoding="utf-8")
print("Script lambda=1 cree ; syntaxe valide ; original inchange.")
