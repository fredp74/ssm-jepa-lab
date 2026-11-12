from pathlib import Path

source = Path("assistant_v10.py").read_text(encoding="utf-8-sig")
marker = '\nif __name__ == "__main__":'
if source.count(marker) != 1:
    raise SystemExit("Point d'entree inattendu.")

old = "    records, reserved = answers(args.test)"
new = old + '''
    for extra in (
        "instructions_v11/validation.jsonl",
        "instructions_v12/repetition_test.jsonl",
    ):
        _, values = answers(Path(extra))
        reserved.update(values)
'''
if source.count(old) != 1:
    raise SystemExit("Structure train inattendue.")
source = source.replace(old, new, 1)

# Les empreintes empechent une reprise avec des tests modifies.
old = '        "trainer_sha256": core.fingerprint(__file__),'
new = old + '''
        "comparison_tests_sha256": {
            name: core.fingerprint(Path(name))
            for name in (
                "instructions_v11/validation.jsonl",
                "instructions_v12/repetition_test.jsonl",
            )
        },
'''
if source.count(old) != 1:
    raise SystemExit("Signature inattendue.")
source = source.replace(old, new, 1)

wrapper = '''
ORIGINAL_FRESH = fresh
REPETITION_ARM = {treatment}

def fresh(rng, used, reserved, maximum, task="copie", length=None):
    selected = False
    if task == "copie" and length is not None and length >= 6:
        selected = rng.random() < 0.5

    if not REPETITION_ARM or not selected:
        return ORIGINAL_FRESH(
            rng, used, reserved, maximum, task, length
        )

    for attempt in range(100000):
        letters = rng.sample(ALPHABET, length)
        position = rng.randrange(length - 1)
        letters[position + 1] = letters[position]
        payload = "".join(letters)
        if payload not in reserved:
            used.add(payload)
            return record(payload, rng.randrange(4), "copie", rng, reserved)
    raise RuntimeError("Generation de doubles caracteres epuisee.")
'''

outputs = [
    (Path("experiment_v13_control.py"), False),
    (Path("experiment_v13_repetition.py"), True),
]
if any(path.exists() for path, _ in outputs):
    raise SystemExit("Un programme v13 existe deja : aucun remplacement.")

prepared = []
for path, treatment in outputs:
    result = source.replace(
        marker,
        "\n" + wrapper.format(treatment=repr(treatment)) + marker,
        1,
    )
    compile(result, str(path), "exec")
    prepared.append((path, result))

for path, result in prepared:
    path.write_text(result, encoding="utf-8")
    print(f"Cree : {path}")
