from pathlib import Path

src = Path("lm_reservoir.py").read_text(encoding="utf-8-sig")

patches = [
    (
        'ROOT = Path("lm_reservoir_v1")',
        'ROOT = Path("lm_reservoir_protected_v1")'
    ),
    (
        'CAPACITY = 4096',
        'CAPACITY = 3072\nPROTECTED = 1024'
    ),
    (
        '        return dict(zip("ABC", counts))',
        '        counts[0] += PROTECTED\n'
        '        return dict(zip("ABC", counts))'
    ),
    (
        '    replay_rng = torch.Generator().manual_seed(SEED+100_000)',
        '    replay_rng = torch.Generator().manual_seed(SEED+100_000)\n'
        '    protected_rng = torch.Generator().manual_seed(SEED+400_000)'
    ),
    (
        '        replay = reservoir.sample(BR, replay_rng)',
        '        protected = from_pool(pool_a[:PROTECTED], 1, protected_rng)\n'
        '        replay = torch.cat([\n'
        '            protected, reservoir.sample(BR-1, replay_rng)\n'
        '        ])'
    ),
    (
        'reservoir = Reservoir(pool_a, SEED+300_000)',
        'reservoir = Reservoir(pool_a[PROTECTED:], SEED+300_000)'
    ),
]

for old, new in patches:
    if src.count(old) != 1:
        raise SystemExit(
            f"Creation interrompue : motif absent ou multiple : {old!r}"
        )
    src = src.replace(old, new, 1)

compile(src, "lm_reservoir_protected.py", "exec")
Path("lm_reservoir_protected.py").write_text(src, encoding="utf-8")
print("Script cree ; syntaxe valide ; original inchange.")
