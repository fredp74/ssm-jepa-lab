import ast
from pathlib import Path

functions = r'''
def record(payload, template, task="copie", rng=None, reserved=None):
    values = [payload]
    if task == "copie":
        question = [
            f"Recopie uniquement le texte entre crochets : [{payload}].",
            f"Reponds seulement avec le contenu de [{payload}].",
            f"Copie sans commentaire : [{payload}].",
            f"Quel texte est entre crochets ? [{payload}] Reponds seulement avec ce texte.",
        ][template]
    else:
        if task != "extraction" or rng is None:
            raise ValueError("Tache ou generateur invalide")
        reserved = set() if reserved is None else reserved
        for _ in range(2):
            for attempt in range(10000):
                word = "".join(
                    rng.choice(ALPHABET)
                    for _ in range(rng.randint(2, 5))
                )
                if word not in reserved and word not in values:
                    values.append(word)
                    break
            else:
                raise RuntimeError("Generation des distracteurs epuisee")
        fields = [
            f"cible={payload}",
            f"gauche={values[1]}",
            f"droite={values[2]}",
        ]
        rng.shuffle(fields)
        text = " ; ".join(fields)
        question = [
            f"Extrais uniquement la valeur du champ cible : {text}.",
            f"Donne seulement la valeur de cible dans : {text}.",
            f"Lis ces champs : {text}. Reponds uniquement avec la valeur de cible.",
            f"Quelle est la valeur de cible ? {text}. Sans commentaire.",
        ][template]
    return {
        "task": f"{task}_L{len(payload)}",
        "source_values": values,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": payload},
        ]
    }


def answers(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text("utf-8-sig").splitlines()
        if line.strip()
    ]
    values = {
        value for row in rows
        for value in row.get(
            "source_values", [row["messages"][-1]["content"]]
        )
    }
    return rows, values


def fresh(rng, used, reserved, maximum, task="copie"):
    for _ in range(10000):
        payload = "".join(
            rng.choice(ALPHABET)
            for _ in range(rng.randint(2, maximum))
        )
        if payload not in used and payload not in reserved:
            used.add(payload)
            return record(payload, rng.randrange(4), task, rng, reserved)
    raise RuntimeError("Generation sans doublon epuisee")


def maximum_length(step, steps):
    return 8


def prepare(path, base):
    if path.exists():
        raise FileExistsError("Test deja present : aucun remplacement")
    state = core.load(base)
    if "stream_used" not in state:
        raise ValueError("Utiliser assistant_stream_v05.pt termine")
    phase = state.get("sft")
    if not phase or phase["step"] < phase["config"]["steps"]:
        raise ValueError("Terminer la phase v0.5 avant cette preparation")

    _, old_test = answers(Path("instructions_v05/test.jsonl"))
    excluded = set(state["stream_used"]) | old_test
    rng = random.Random(61006)
    rows = []

    for task, maximum in (("copie", 8), ("extraction", 5)):
        for length in range(2, maximum+1):
            for index in range(16):
                for attempt in range(10000):
                    payload = "".join(
                        rng.choice(ALPHABET) for _ in range(length)
                    )
                    if payload not in excluded:
                        break
                else:
                    raise RuntimeError("Generation du test epuisee")
                row = record(payload, index % 4, task, rng, excluded)
                excluded.update(row["source_values"])
                rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
        encoding="utf-8"
    )
    print(
        f"Test fixe : {path} | 112 copies + 64 extractions | "
        "16 exemples par tache/longueur"
    )


def audit_add(state, row):
    payload = row["messages"][-1]["content"]
    bucket = state["stream_audit"].setdefault(row["task"], [])
    bucket.append({
        "rank": hashlib.sha256(payload.encode()).hexdigest(),
        "record": row
    })
    bucket.sort(key=lambda item: item["rank"])
    del bucket[16:]


def smoke():
    import re
    rng = random.Random(6)
    used, reserved = set(), {"aa", "bb", "12"}
    rows = [
        fresh(rng, used, reserved, 5, "extraction")
        for _ in range(40)
    ]
    assert len(used) == 40 and not used & reserved
    positions = set()
    for row in rows:
        question = row["messages"][0]["content"]
        expected = row["messages"][1]["content"]
        assert re.search(
            r"cible=([a-z0-9]+)", question
        ).group(1) == expected
        assert len(set(row["source_values"])) == 3
        assert not set(row["source_values"]) & reserved
        positions.add(
            re.findall(r"(cible|gauche|droite)=", question).index("cible")
        )
    assert positions == {0, 1, 2}
    saved = rng.getstate()
    other = random.Random()
    other.setstate(saved)
    copied = set(used)
    assert fresh(
        rng, used, reserved, 5, "extraction"
    ) == fresh(other, copied, reserved, 5, "extraction")
    extension.smoke()
    print(
        "MIXED SMOKE OK : champs, distracteurs, positions variables, "
        "exclusion test et RNG."
    )
'''

source = Path("assistant_v05.py").read_text(encoding="utf-8-sig")
destination = Path("assistant_v06.py")
if destination.exists():
    raise SystemExit("assistant_v06.py existe deja : aucun remplacement")

patches = [
    (
        'instructions_v05/test.jsonl',
        'instructions_v06/test.jsonl'
    ),
    (
        '"objective": "fresh_copy_curriculum_v1"',
        '"objective": "fresh_copy_extraction_v1"'
    ),
    (
        'records, reserved = answers(args.test)',
        'records, reserved = answers(args.test)\n'
        '    _, prior_test = answers(Path("instructions_v05/test.jsonl"))\n'
        '    reserved |= prior_test'
    ),
    (
        '"test_sha256": core.fingerprint(args.test),',
        '"test_sha256": core.fingerprint(args.test),\n'
        '        "prior_test_sha256": core.fingerprint(Path("instructions_v05/test.jsonl")),'
    ),
    (
        'records = [fresh(rng, used, reserved, maximum) for _ in range(4)]',
        'records = ([fresh(rng, used, reserved, 8, "copie") for _ in range(2)]\n'
        '               + [fresh(rng, used, reserved, 5, "extraction") for _ in range(2)])'
    ),
    (
        'if state["arch"] != extension.ARCH or state.get("sft"):',
        'if state["arch"] != extension.ARCH or (state.get("sft") and state["sft"]["step"] < state["sft"]["config"]["steps"]):'
    ),
    (
        'state["stream_used"] = []',
        'state["stream_used"] = list(state.get("stream_used", []))\n'
        '        state["stream_initial_count"] = len(state["stream_used"])'
    ),
    (
        'sorted(state["stream_audit"], key=int)',
        'sorted(state["stream_audit"])'
    ),
    (
        'longueur 2..{maximum}',
        'copie 2..8, extraction 2..5'
    ),
    (
        '{len(used)} chaines distinctes',
        "{len(used)-state['stream_initial_count']} nouvelles cibles"
    ),
    (
        'p = sub.add_parser("prepare")',
        'p = sub.add_parser("prepare")\n'
        '    p.add_argument("--base", type=Path, default=Path("runs/assistant_stream_v05.pt"))'
    ),
    ('default=1200)', 'default=800)'),
    ('prepare(args.out)', 'prepare(args.out, args.base)'),
]

for old, new in patches:
    expected = 2 if old == 'instructions_v05/test.jsonl' else 1
    if source.count(old) != expected:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new)

for replacement in ast.parse(functions).body:
    if not isinstance(replacement, ast.FunctionDef):
        continue
    tree = ast.parse(source)
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == replacement.name
    ]
    if len(matches) != 1:
        raise SystemExit(f"Fonction introuvable : {replacement.name}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    body = ast.get_source_segment(functions, replacement)
    source = (
        "".join(lines[:node.lineno-1]) + body + "\n"
        + "".join(lines[node.end_lineno:])
    )

compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print("assistant_v06.py cree ; syntaxe valide ; anciens scripts conserves.")
