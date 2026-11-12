import json
from pathlib import Path

root = Path("autonomous_lab")
plan_path = root / "replication_plan.json"
if plan_path.exists():
    raise SystemExit("Plan deja present : aucun remplacement.")

prepared = []
plan = {
    "purpose": "Replicate frozen question-selection experiment",
    "changes": "Model initialization and training sampling seeds only",
    "replications": [],
}

for number, seeds in (
    ("02", [62101, 62102, 62103]),
    ("03", [62201, 62202, 62203]),
):
    jobs = [
        (
            "train_partial_ensemble.py",
            f"train_partial_ensemble_{number}.py",
            [
                (
                    "seeds = [62001, 62002, 62003]",
                    f"seeds = {seeds}",
                ),
                (
                    "runs/partial_context_ensemble_01",
                    f"runs/partial_context_ensemble_{number}",
                ),
            ],
        ),
        (
            "evaluate_ensemble_selection.py",
            f"evaluate_ensemble_selection_{number}.py",
            [
                (
                    "runs/partial_context_ensemble_01",
                    f"runs/partial_context_ensemble_{number}",
                ),
                (
                    "runs/ensemble_selection_01",
                    f"runs/ensemble_selection_{number}",
                ),
            ],
        ),
        (
            "exhaustive_ensemble_random.py",
            f"exhaustive_ensemble_random_{number}.py",
            [
                (
                    "runs/partial_context_ensemble_01",
                    f"runs/partial_context_ensemble_{number}",
                ),
                (
                    "runs/exhaustive_ensemble_random_01",
                    f"runs/exhaustive_ensemble_random_{number}",
                ),
                (
                    "runs/ensemble_selection_01",
                    f"runs/ensemble_selection_{number}",
                ),
            ],
        ),
    ]

    for original, destination, patches in jobs:
        path = root / destination
        if path.exists():
            raise SystemExit(f"Programme deja present : {path}")

        source = (root / original).read_text(encoding="utf-8-sig")
        for old, new in patches:
            if source.count(old) != 1:
                raise SystemExit(
                    f"Structure inattendue dans {original} : {old!r}"
                )
            source = source.replace(old, new, 1)

        compile(source, str(path), "exec")
        prepared.append((path, source))

    plan["replications"].append({
        "number": number,
        "seeds": seeds,
    })

for path, source in prepared:
    path.write_text(source, encoding="utf-8")
    print(f"Cree : {path}")

plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
print(f"Plan enregistre : {plan_path}")
