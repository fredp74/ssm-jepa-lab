from pathlib import Path

src = Path("lm_jepa.py").read_text(encoding="utf-8-sig")

patches = [
    (
        'ROOT = Path("lm_jepa_v1")',
        'ROOT = Path("lm_jepa_block32_v1")'
    ),
    (
        'K, WEIGHT, EMA = 4, 0.1, 0.99',
        'K, WEIGHT, EMA = 32, 1.0, 0.99\nSEEDS = (1000,)'
    ),
    (
        '''        prediction = predictor(h[:, :-K])
        jepa = F.mse_loss(prediction, ht[:, K:])''',
        '''        # 256 positions = 8 blocs de 32.
        # Etats aux positions 31,63,...,223 : fins de contexte.
        context = h[:, K-1:L-K:K]
        target_blocks = ht.reshape(ht.size(0), L//K, K, 192).mean(2)
        future = target_blocks[:, 1:]
        prediction = predictor(context)
        jepa = F.mse_loss(prediction, future)

        if (step+1) % 500 == 0:
            with torch.no_grad():
                copied = target_blocks[:, :-1]
                mean = future.mean(dim=(0, 1), keepdim=True)

                # Permutation sans melanger C, A et B.
                shuffled = torch.cat([
                    future[:B].roll(1, dims=0),
                    future[B:B+2].roll(1, dims=0),
                    future[B+2:].roll(1, dims=0),
                ])
                variance = F.mse_loss(mean.expand_as(future), future)
                copy_error = F.mse_loss(copied, future)
                shuffle_error = F.mse_loss(prediction, shuffled)
                denom = max(variance.item(), 1e-12)
                print(
                    f"BLOCS seed={seed} pas={step+1} | "
                    f"JEPA/variance={jepa.item()/denom:.4f} | "
                    f"copie/variance={copy_error.item()/denom:.4f} | "
                    f"melange/variance={shuffle_error.item()/denom:.4f}",
                    flush=True
                )'''
    ),
    (
        '''        "weight": WEIGHT, "ema": EMA,''',
        '''        "weight": WEIGHT, "ema": EMA,
        "objective": "next_block_mean", "block_size": K,'''
    ),
]

for old, new in patches:
    if src.count(old) != 1:
        raise SystemExit(
            f"Creation interrompue : motif absent ou multiple : {old!r}"
        )
    src = src.replace(old, new, 1)

compile(src, "lm_jepa_block32.py", "exec")
Path("lm_jepa_block32.py").write_text(src, encoding="utf-8")
print("Script blocs cree ; syntaxe valide ; original inchange.")
