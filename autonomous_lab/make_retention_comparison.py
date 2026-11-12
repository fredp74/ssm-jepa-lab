from pathlib import Path

source = Path(
    "autonomous_lab/compare_observation_curriculum.py"
).read_text(encoding="utf-8-sig")
destination = Path("autonomous_lab/compare_retention.py")
if destination.exists():
    raise SystemExit("Programme deja present : aucun remplacement.")

helper = '''
@torch.no_grad()
def recency(model, rules):
    model.eval()
    inputs = torch.tensor(INPUTS)
    generator = torch.Generator().manual_seed(52999)
    correct = [0] * 8
    totals = [0] * 8

    for rule in rules:
        ids = torch.tensor([rule])
        for _ in range(24):
            order = torch.randperm(8, generator=generator)
            x = inputs[order][None]
            y = environment(ids, x)
            answers = model(x, y, x).argmax(-1)[0]
            for position in range(8):
                correct[position] += int(
                    answers[position] == y[0, position]
                )
                totals[position] += 1

    return {
        "correct_by_position": correct,
        "total_by_position": totals,
    }

'''

patches = [
    (
        'Path("autonomous_lab/runs/observation_curriculum_01")',
        'Path("autonomous_lab/runs/retention_comparison_01")',
    ),
    (
        'for arm in ("fixed", "variable"):',
        'for arm in ("standard", "retention"):',
    ),
    (
        '            model = MetaSSM()',
        '            model = MetaSSM()\n'
        '            if arm == "retention":\n'
        '                with torch.no_grad():\n'
        '                    model.decay.fill_(2.9444389791664403)',
    ),
    (
        '                if arm == "fixed":',
        '                if False:  # Les deux branches utilisent le protocole variable.',
    ),
    (
        '                "test": evaluate(model, TEST_RULES),',
        '                "test": evaluate(model, TEST_RULES),\n'
        '                "train_recency": recency(model, TRAIN_RULES),\n'
        '                "test_recency": recency(model, TEST_RULES),\n'
        '                "final_decay": model.decay.detach().sigmoid().tolist(),',
    ),
    (
        '                        "train_rules": TRAIN_RULES,',
        '                        "initial_decay": {"standard": 0.5, "retention": 0.95},\n'
        '                        "decay_trainable": True,\n'
        '                        "train_rules": TRAIN_RULES,',
    ),
    (
        'def main():',
        helper + '\ndef main():',
    ),
]

for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print(f"Programme cree : {destination}")
