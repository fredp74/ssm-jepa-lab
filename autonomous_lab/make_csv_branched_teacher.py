from pathlib import Path

source = Path(
    "autonomous_lab/csv_symbolic_teacher.py"
).read_text(encoding="utf-8-sig")
destination = Path("autonomous_lab/csv_branched_teacher.py")
if destination.exists():
    raise SystemExit("Programme deja present : aucun remplacement.")

patches = [
    (
        "import json",
        "import json\nimport random",
    ),
    (
        "csv_symbolic_teacher_01",
        "csv_branched_teacher_01",
    ),
    (
        "    for environment_id, environment in enumerate(classes):",
        "    for rollout_id in range(len(classes)*4):\n"
        "        environment_id = rollout_id // 4\n"
        "        repetition = rollout_id % 4\n"
        "        environment = classes[environment_id]\n"
        "        rng = random.Random(98001000 + rollout_id)",
    ),
    (
        "            selected = best_actions[0]",
        "            selected = rng.choice(best_actions)",
    ),
    (
        '            "environment_class_id": environment_id,',
        '            "environment_class_id": environment_id,\n'
        '            "tie_breaking_repetition": repetition,',
    ),
    (
        '        "observable_known_classes": sum(',
        '        "known_trajectories": sum(',
    ),
    (
        '        "detectable_classes": sum(',
        '        "detectable_trajectories": sum(',
    ),
    (
        '        "detected_unknown_classes": sum(',
        '        "detected_unknown_trajectories": sum(',
    ),
    (
        '        "decision_steps": sum(len(row["steps"]) for row in trajectories),',
        '        "decision_steps": sum(len(row["steps"]) for row in trajectories),\n'
        '        "distinct_decision_states": len({\n'
        '            (tuple(step["tested_ids_before"]),\n'
        '             tuple(step["known_candidates_before"]),\n'
        '             step["remaining_calls"], step["remaining_cells"],\n'
        '             step["remaining_text_bytes"])\n'
        '            for row in trajectories for step in row["steps"]\n'
        '        }),',
    ),
    (
        '                "globally_optimal_policy": False,',
        '                "globally_optimal_policy": False,\n'
        '                "trajectories_per_observable_class": 4,\n'
        '                "tie_breaking": "uniform_random_among_best_actions",\n'
        '                "tie_breaking_seed_base": 98001000,\n'
        '                "student_induced_states_collected": False,',
    ),
]

for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print(f"Programme cree : {destination}")
