from pathlib import Path

source = Path(
    "autonomous_lab/evaluate_csv_policy_rollout.py"
).read_text(encoding="utf-8-sig")
destination = Path("autonomous_lab/evaluate_csv_teacher_branch.py")
if destination.exists():
    raise SystemExit("Programme deja present : aucun remplacement.")

old_choice = '''                        action = max(
                            feasible, key=lambda i: (*score(i), -i)
                        )'''

new_choice = old_choice + '''
                        if (
                            policy == "symbolic_force18"
                            and used == [39, 42, 7, 43]
                            and 18 in feasible
                            and action == 11
                            and score(18) == score(action)
                        ):
                            action = 18
                            forced_actions += 1'''

patches = [
    (
        'runs/csv_policy_rollout_01',
        'runs/csv_teacher_branch_01',
    ),
    (
        'policies = ["fixed", "random", "lookup", "symbolic"] + list(models)',
        'policies = ["symbolic", "symbolic_force18"]',
    ),
    (
        '                unfamiliar = fallbacks = 0',
        '                unfamiliar = fallbacks = 0\n'
        '                forced_actions = 0',
    ),
    (old_choice, new_choice),
    (
        '                    if policy == "symbolic":',
        '                    if policy in ("symbolic", "symbolic_force18"):',
    ),
    (
        '                    "lookup_fallbacks": fallbacks,',
        '                    "lookup_fallbacks": fallbacks,\n'
        '                    "forced_actions": forced_actions,',
    ),
    (
        '        "evaluation_updates": 0,',
        '        "evaluation_updates": 0,\n'
        '        "intervention": "force_test_18_instead_of_tied_test_11",\n'
        '        "intervention_prefix": [39, 42, 7, 43],\n'
        '        "recompute_teacher_after_intervention": True,\n'
        '        "previously_consulted_diagnostic_cases": True,',
    ),
]

for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print(f"Programme cree : {destination}")
