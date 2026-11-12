import ast
from pathlib import Path

functions = r'''
def record(payload, template, task="copie", rng=None, reserved=None, last_style=None):
    if task != "dialogue":
        return single_record(payload, template, task, rng, reserved)
    last_style = rng.randrange(4) if last_style is None else last_style
    first = [
        f"Retiens ce code : {payload}. Reponds seulement OK.",
        f"Memorise ce code : {payload}. Confirme uniquement par OK.",
        f"Le code a retenir est {payload}. Reponds uniquement OK.",
        f"Garde en memoire le code {payload}. Ecris seulement OK.",
    ][template]
    last = [
        "Quel code viens-je de donner ? Reponds seulement avec ce code.",
        "Redonne uniquement le code indique au message precedent.",
        "Quel est le code memorise ? Sans commentaire.",
        "Reponds seulement avec le code que je t'ai donne.",
    ][last_style]
    return {
        "task": f"dialogue_L{len(payload)}",
        "source_values": [payload],
        "first_style": template,
        "last_style": last_style,
        "messages": [
            {"role": "user", "content": first},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": last},
            {"role": "assistant", "content": payload},
        ]
    }


def prepare(path, base):
    if path.exists():
        raise FileExistsError("Test deja present : aucun remplacement")
    state = core.load(base)
    phase = state.get("sft")
    if not phase or phase["step"] < phase["config"]["steps"]:
        raise ValueError("Utiliser le checkpoint termine de la v0.7")
    excluded = set(state["stream_used"])
    for name in (
        "instructions_v05/test.jsonl",
        "instructions_v06/test.jsonl",
        "instructions_v07/fresh_validation.jsonl",
    ):
        _, values = answers(Path(name))
        excluded.update(values)

    rng = random.Random(81009)
    targets = []
    for task, maximum in (
        ("copie", 8), ("extraction", 5), ("dialogue", 4)
    ):
        for length in range(2, maximum+1):
            for index in range(16):
                for attempt in range(10000):
                    payload = "".join(
                        rng.choice(ALPHABET) for _ in range(length)
                    )
                    if payload not in excluded:
                        break
                else:
                    raise RuntimeError(
                        f"Pas assez de cibles de longueur {length}"
                    )
                excluded.add(payload)
                targets.append((task, payload, index))

    rows = []
    for task, payload, index in targets:
        row = record(
            payload, index % 4, task, rng, excluded, (index//4) % 4
        )
        excluded.update(row["source_values"])
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
        encoding="utf-8"
    )
    print(
        f"Test fixe : {path} | "
        "112 copies + 64 extractions + 48 dialogues"
    )


@torch.no_grad()
def history_controls(args):
    model = extension.factory(core.load(args.run)).eval()
    records, _ = answers(args.data)
    groups = {}
    for row in records:
        if row["task"].startswith("dialogue_"):
            key = (
                len(row["messages"][-1]["content"]),
                row["first_style"]
            )
            groups.setdefault(key, []).append(row)

    totals = {}
    for (length, style), rows in sorted(groups.items()):
        if len(rows) < 2:
            raise ValueError(
                "Au moins deux dialogues par longueur/style pour permuter"
            )
        stats = totals.setdefault(length, [0, 0, 0, 0, 0])
        for i, row in enumerate(rows):
            other = rows[(i+1) % len(rows)]
            question = row["messages"][2]["content"]
            expected = row["messages"][3]["content"]
            transplanted = other["messages"][3]["content"]
            histories = [
                [(row["messages"][0]["content"], "OK")],
                [],
                [(other["messages"][0]["content"], "OK")],
            ]
            outcomes = []
            for history in histories:
                prefix, _ = app.conversation_prefix(
                    history, question, 512
                )
                result, ended = app.stream_bytes(
                    model, prefix, 96, 0, 40, 42, app.END.encode()
                )
                outcomes.append((
                    result.decode("utf-8", errors="replace").strip(),
                    ended
                ))
            stats[0] += 1
            for j in range(3):
                stats[j+1] += int(
                    outcomes[j][1] and outcomes[j][0] == expected
                )
            stats[4] += int(
                outcomes[2][1] and outcomes[2][0] == transplanted
            )

    for length, stats in sorted(totals.items()):
        n, normal, absent, wrong_old, follows_new = stats
        print(
            f"HISTOIRE L{length} | n={n} | "
            f"reference_OK={normal}/{n} | "
            f"sans_histoire={absent}/{n} | "
            f"permute_repond_ancien={wrong_old}/{n} | "
            f"permute_suit_nouveau={follows_new}/{n}",
            flush=True
        )


def smoke():
    rng = random.Random(8)
    row = record("z9q", 0, "dialogue", rng)
    assert "z9q" in row["messages"][0]["content"]
    assert "z9q" not in row["messages"][2]["content"]
    assert row["messages"][1]["content"] == "OK"

    x, y = app.instruction_batch([app.encoded(row)], [0])
    content, boundary, tail = masks3(y)
    assert bytes(y[content].tolist()).decode() == "OKz9q"
    assert boundary.sum().item() == 2
    assert tail.sum().item() == 2*(len(app.END.encode())-1)

    extension.smoke()
    print(
        "DIALOGUE SMOKE OK : format deux tours, masques "
        "et code absent du dernier message."
    )
'''

source = Path("assistant_v07.py").read_text(encoding="utf-8-sig")
destination = Path("assistant_v08.py")
if destination.exists():
    raise SystemExit("assistant_v08.py existe deja : aucun remplacement")

patches = [
    (
        'instructions_v06/test.jsonl',
        'instructions_v08/test.jsonl',
        2
    ),
    (
        '"objective": "balanced_copy_extraction_boundary_v1"',
        '"objective": "balanced_copy_extract_dialogue_v1"',
        1
    ),
    ('def record(', 'def single_record(', 1),
    (
        'records, reserved = answers(args.test)',
        'records, reserved = answers(args.test)\n'
        '    for name in ("instructions_v06/test.jsonl", "instructions_v07/fresh_validation.jsonl"):\n'
        '        _, values = answers(Path(name))\n'
        '        reserved.update(values)',
        1
    ),
    (
        '"prior_test_sha256": core.fingerprint(Path("instructions_v05/test.jsonl")),',
        '"prior_test_sha256": core.fingerprint(Path("instructions_v05/test.jsonl")),\n'
        '        "additional_tests_sha256": {name: core.fingerprint(Path(name)) for name in ("instructions_v06/test.jsonl", "instructions_v07/fresh_validation.jsonl")},',
        1
    ),
    (
        'copie 2..8, extraction 2..5',
        'copie/extraction/dialogue',
        1
    ),
    (
        'sub.add_parser("smoke")',
        'sub.add_parser("smoke")\n'
        '    p = sub.add_parser("prepare")\n'
        '    p.add_argument("--out", type=Path, default=Path("instructions_v08/test.jsonl"))\n'
        '    p.add_argument("--base", type=Path, default=Path("runs/assistant_balanced_v07.pt"))',
        1
    ),
    (
        'p.add_argument("--limit-per-task", type=int, default=16)',
        'p.add_argument("--limit-per-task", type=int, default=16)\n'
        '    p.add_argument("--history-controls", action="store_true")',
        1
    ),
    (
        'elif args.command == "eval":',
        'elif args.command == "prepare":\n'
        '        prepare(args.out, args.base)\n'
        '    elif args.command == "eval":',
        1
    ),
    (
        'app.evaluate(args)',
        'app.evaluate(args)\n'
        '        if args.history_controls:\n'
        '            history_controls(args)',
        1
    ),
    ('default=400)', 'default=800)', 1),
]
for old, new, count in patches:
    if source.count(old) != count:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new)

tree = ast.parse(source)
node = next(
    n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "update"
)
old_update = ast.get_source_segment(source, node)
marker = "    rows = [app.encoded(row) for row in records]"
if old_update.count(marker) != 1:
    raise SystemExit("Structure update inattendue")
tail = marker + old_update.split(marker, 1)[1]
new_update = '''def update(model, opt, state, rng, replay_rng, used, reserved):
    step = state["sft"]["step"]
    config = state["sft"]["config"]
    records = [fresh(rng, used, reserved, 8, "copie", 2+step%7)]
    records += [fresh(rng, used, reserved, 5, "extraction", 2+step%4)]
    records += [fresh(rng, used, reserved, 4, "dialogue", 2+(2*step+j)%3) for j in range(2)]
'''+tail
source = source.replace(old_update, new_update, 1)

to_add = []
for replacement in ast.parse(functions).body:
    if not isinstance(replacement, ast.FunctionDef):
        continue
    tree = ast.parse(source)
    matches = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == replacement.name
    ]
    first = min(
        [replacement.lineno]
        + [d.lineno for d in replacement.decorator_list]
    )-1
    body = "".join(
        functions.splitlines(keepends=True)[first:replacement.end_lineno]
    )
    if not matches:
        to_add.append(body)
        continue
    if len(matches) != 1:
        raise SystemExit(f"Fonction multiple : {replacement.name}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    source = (
        "".join(lines[:node.lineno-1])+body+"\n"
        +"".join(lines[node.end_lineno:])
    )

marker = '\nif __name__ == "__main__":'
if source.count(marker) != 1:
    raise SystemExit("Point d'entree inattendu")
source = source.replace(
    marker, "\n"+"\n\n".join(to_add)+"\n"+marker, 1
)
compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print("assistant_v08.py cree ; syntaxe valide ; anciens fichiers conserves.")
