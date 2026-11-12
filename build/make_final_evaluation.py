from pathlib import Path

extra_checkpoints = "".join(
    f'    "runs/experiment_v13_{arm}{suffix}.pt",\n'
    for suffix in ("", "_130014", "_130015")
    for arm in ("control", "repetition")
)

jobs = [
    (
        "prepare_validation_v11.py",
        "prepare_final_general.py",
        "instructions_v11/validation.jsonl",
        "final_evaluation/general.jsonl",
        "random.Random(111201)",
        "random.Random(26011201)",
    ),
    (
        "prepare_repetition_test.py",
        "prepare_final_repetitions.py",
        "instructions_v12/repetition_test.jsonl",
        "final_evaluation/repetitions.jsonl",
        "random.Random(120012)",
        "random.Random(26011202)",
    ),
]

prepared = []
for original, destination, old_path, new_path, old_seed, new_seed in jobs:
    destination = Path(destination)
    if destination.exists() or Path(new_path).exists():
        raise SystemExit(f"Destination deja presente : {destination} ou {new_path}")

    source = Path(original).read_text(encoding="utf-8-sig")
    checkpoint = '    "runs/assistant_longcopy_v10.pt",\n'

    for marker in (old_path, old_seed, checkpoint):
        if source.count(marker) != 1:
            raise SystemExit(f"Structure inattendue dans {original} : {marker!r}")

    source = source.replace(old_path, new_path, 1)
    source = source.replace(old_seed, new_seed, 1)
    source = source.replace(checkpoint, checkpoint + extra_checkpoints, 1)

    compile(source, str(destination), "exec")
    prepared.append((destination, source))

for destination, source in prepared:
    destination.write_text(source, encoding="utf-8")
    print(f"Cree : {destination}")
