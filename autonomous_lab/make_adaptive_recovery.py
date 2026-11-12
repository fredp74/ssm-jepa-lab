from pathlib import Path

source = Path("autonomous_lab/compare_change_recovery.py").read_text(
    encoding="utf-8-sig"
)
destination = Path("autonomous_lab/compare_adaptive_recovery.py")
if destination.exists():
    raise SystemExit("Programme deja present : aucun remplacement.")

old_cases = '''    cases = (
        [("single_flip", (i,)) for i in range(32)]
        + [("double_flip", pair) for pair in pairs]
        + [("inversion", tuple(range(32)))]
    )'''

new_cases = '''    cases = (
        [("single_flip", (i,)) for i in range(32)]
        + [("double_flip", pair) for pair in pairs]
        + [
            (f"flip_{size}", tuple(sorted(rng.sample(range(32), size))))
            for size in (4, 8, 16)
            for _ in range(32)
        ]
        + [("inversion", tuple(range(32)))]
    )'''

old_prediction = '''                    prediction = (
                        reset.prediction(index)[0]
                        if method == "reset"
                        else historical[index]
                    )'''

new_prediction = '''                    if method == "reset":
                        prediction = reset.prediction(index)[0]
                    elif method == "local_history":
                        prediction = historical[index]
                    else:
                        # Ne pas compter la contradiction ayant declenche
                        # la detection comme un controle aleatoire.
                        checked = [i for i in fresh if i != detected_index]
                        mismatches = sum(
                            historical[i] != FIRST[i] for i in checked
                        )
                        if len(checked) < 4:
                            prediction = None
                        elif 2*mismatches >= len(checked):
                            prediction = reset.prediction(index)[0]
                        else:
                            prediction = historical[index]'''

patches = [
    (
        'Path("autonomous_lab/runs/change_recovery_01")',
        'Path("autonomous_lab/runs/adaptive_recovery_01")',
        1,
    ),
    (old_cases, new_cases, 1),
    (
        '("reset", "local_history")',
        '("reset", "local_history", "adaptive")',
        2,
    ),
    (old_prediction, new_prediction, 1),
    (
        '("single_flip", "double_flip", "inversion")',
        '("single_flip", "double_flip", "flip_4", "flip_8", "flip_16", "inversion")',
        1,
    ),
    ('random.Random(94001)', 'random.Random(94101)', 1),
    ('random.Random(94001000+case_id)', 'random.Random(94101000+case_id)', 1),
    (
        '        "seed": 94001,',
        '        "seed": 94101,\n'
        '        "intermediate_flip_counts": [4, 8, 16],\n'
        '        "sampled_cases_per_intermediate_count": 32,\n'
        '        "adaptive_minimum_checks": 4,\n'
        '        "adaptive_reset_threshold": 0.5,\n'
        '        "adaptive_early_behavior": "abstain_on_unreobserved_inputs",\n'
        '        "adaptive_policy_is_handwritten": True,',
        1,
    ),
]

for old, new, count in patches:
    if source.count(old) != count:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new)

compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print(f"Programme cree : {destination}")
