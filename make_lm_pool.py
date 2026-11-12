from pathlib import Path

src = Path("lm_buffer.py").read_text(encoding="utf-8-sig")

patches = [
    (
        'args = parser.parse_args()',
        'parser.add_argument("--pool-seed", type=int, required=True)\n'
        'args = parser.parse_args()'
    ),
    (
        'SEEDS = (1000, 1001, 1002)',
        'SEEDS = (1000,)'
    ),
    (
        'CONDS = ("sans replay", "tampon256", "tampon4096", "replay complet")',
        'CONDS = ("tampon4096",)'
    ),
    (
        'pool_gen = torch.Generator().manual_seed(424242)',
        'pool_gen = torch.Generator().manual_seed(args.pool_seed)'
    ),
    (
        'out = Path(f"lm_buffer_v1_{args.model}.csv")',
        'out = Path(f"lm_pool_v1_{args.model}_{args.pool_seed}.csv")'
    ),
    (
        'print("\\nMOYENNE +/- ECART-TYPE entre graines de phase B")',
        'print(f"\\nTAMPON seed={args.pool_seed} ; phase B seed=1000")'
    ),
]

for old, new in patches:
    if src.count(old) != 1:
        raise SystemExit(
            f"Creation interrompue : motif absent ou multiple : {old!r}"
        )
    src = src.replace(old, new, 1)

compile(src, "lm_pool.py", "exec")
Path("lm_pool.py").write_text(src, encoding="utf-8")
print("lm_pool.py cree ; syntaxe valide ; lm_buffer.py inchange.")
