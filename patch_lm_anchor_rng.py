from pathlib import Path
import py_compile

p = Path("lm_anchor.py")
src = p.read_text(encoding="utf-8-sig")
backup = Path("lm_anchor_before_rng_fix.py")

patches = [
    (
        "HOLD, NEVAL = 500_000, 200",
        "HOLD, NEVAL = 500_000, 200\n"
        "torch.set_num_threads(8)"
    ),
    (
        "def batch(k, n):",
        "def batch(k, n, generator):"
    ),
    (
        "torch.randint(lo, hi - L - 1, (n,))",
        "torch.randint(lo, hi - L - 1, (n,), generator=generator)"
    ),
    (
        "        torch.manual_seed(1000)",
        "        torch.manual_seed(1000)\n"
        "        gen_b = torch.Generator().manual_seed(1000)\n"
        "        gen_a = torch.Generator().manual_seed(1001)"
    ),
    (
        'xb = batch("B", B)',
        'xb = batch("B", B, gen_b)'
    ),
    (
        'xa = batch("A", BR)',
        'xa = batch("A", BR, gen_a)'
    ),
]

for old, new in patches:
    if src.count(old) != 1:
        raise SystemExit(
            f"Patch interrompu : motif absent ou multiple : {old!r}. "
            "Aucun fichier modifie."
        )
    src = src.replace(old, new, 1)

if backup.exists():
    raise SystemExit("Sauvegarde deja presente : aucun fichier modifie.")

backup.write_bytes(p.read_bytes())
p.write_text(src, encoding="utf-8")
py_compile.compile(str(p), doraise=True)
print("Patch applique ; syntaxe Python valide.")
