import ast
from pathlib import Path

functions = r'''
def recall_question(style):
    return [
        "Quel code viens-je de donner ? Reponds seulement avec ce code.",
        "Redonne uniquement le code indique au message precedent.",
        "Quel est le code memorise ? Sans commentaire.",
        "Reponds seulement avec le code que je t'ai donne.",
    ][style]


def record(payload, template, task="copie", rng=None, reserved=None,
           last_style=None, old_code=None, update_style=None):
    if task != "mise_a_jour":
        return previous_record(
            payload, template, task, rng, reserved, last_style
        )
    reserved = set() if reserved is None else reserved
    if old_code is None:
        for attempt in range(10000):
            old_code = "".join(
                rng.choice(ALPHABET) for _ in range(len(payload))
            )
            if old_code != payload and old_code not in reserved:
                break
        else:
            raise RuntimeError("Generation ancien code epuisee")
    if old_code == payload or len(old_code) != len(payload):
        raise ValueError("Deux codes distincts de meme longueur attendus")

    last_style = rng.randrange(4) if last_style is None else last_style
    update_style = (
        rng.randrange(4) if update_style is None else update_style
    )
    initial = previous_record(
        old_code, template, "dialogue", rng, reserved, last_style
    )
    change = [
        f"Remplace ce code par {payload}. Reponds seulement OK.",
        f"Le nouveau code est {payload}. Confirme uniquement par OK.",
        f"Oublie l'ancien code et retiens {payload}. Reponds uniquement OK.",
        f"Mets le code a jour : {payload}. Ecris seulement OK.",
    ][update_style]
    return {
        "task": f"mise_a_jour_L{len(payload)}",
        "source_values": [old_code, payload],
        "old_code": old_code,
        "new_code": payload,
        "update_style": update_style,
        "messages": initial["messages"][:2] + [
            {"role": "user", "content": change},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": recall_question(last_style)},
            {"role": "assistant", "content": payload},
        ]
    }


def missing_record(style, neutral=False, neutral_style=0):
    messages = []
    if neutral:
        first = [
            "Nous allons commencer. Reponds seulement OK.",
            "Je vais poser une question. Confirme uniquement par OK.",
            "Un nouveau dialogue commence. Reponds uniquement OK.",
            "Attendons la suite. Ecris seulement OK.",
        ][neutral_style]
        messages += [
            {"role": "user", "content": first},
            {"role": "assistant", "content": "OK"}
        ]
    messages += [
        {"role": "user", "content": recall_question(style)},
        {"role": "assistant", "content": "INCONNU"}
    ]
    return {
        "task": "inconnu_neutre" if neutral else "inconnu_direct",
        "source_values": [],
        "messages": messages
    }


def audit_add(state, row):
    bucket = state["stream_audit"].setdefault(row["task"], [])
    identity = json.dumps(
        row["messages"], ensure_ascii=False, sort_keys=True
    )
    rank = hashlib.sha256(identity.encode()).hexdigest()
    if any(item["rank"] == rank for item in bucket):
        return
    bucket.append({"rank": rank, "record": row})
    bucket.sort(key=lambda item: item["rank"])
    del bucket[16:]


def prepare(path, base):
    if path.exists():
        raise FileExistsError("Test deja present : aucun remplacement")
    state = core.load(base)
    phase = state.get("sft")
    if not phase or phase["step"] < phase["config"]["steps"]:
        raise ValueError("Terminer la v0.8 avant cette etape")

    existing, held = answers(Path("instructions_v08/test.jsonl"))
    if set(state["stream_used"]) & held:
        raise ValueError(
            "Les anciennes valeurs de test ont ete utilisees comme cibles"
        )
    rows = list(existing)
    rng = random.Random(91009)
    for length in range(2, 5):
        olds = [
            r["messages"][-1]["content"] for r in existing
            if r["task"] == f"copie_L{length}"
        ]
        news = [
            r["messages"][-1]["content"] for r in existing
            if r["task"] == f"dialogue_L{length}"
        ]
        if len(olds) != 16 or len(news) != 16:
            raise ValueError("Jeu v0.8 incompatible")
        rng.shuffle(olds)
        rng.shuffle(news)
        for i, (old, new) in enumerate(zip(olds, news)):
            rows.append(record(
                new, i % 4, "mise_a_jour", rng, held,
                (i+i//4) % 4, old, (i//4) % 4
            ))

    rows += [missing_record(style) for style in range(4)]
    rows += [
        missing_record(style, True, neutral_style)
        for neutral_style in range(4) for style in range(4)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
        encoding="utf-8"
    )
    print(
        f"Validation : {path} | "
        "224 anciens cas + 48 mises a jour + 20 absences"
    )


@torch.no_grad()
def update_controls(args):
    model = extension.factory(core.load(args.run)).eval()
    records, _ = answers(args.data)
    groups = {}
    for row in records:
        if row["task"].startswith("mise_a_jour_"):
            key = (len(row["new_code"]), row["update_style"])
            groups.setdefault(key, []).append(row)

    totals = {}
    for (length, style), rows in sorted(groups.items()):
        if len(rows) < 2:
            raise ValueError("Groupe trop petit pour permuter")
        stats = totals.setdefault(length, [0, 0, 0, 0, 0])
        for i, row in enumerate(rows):
            other = rows[(i+1) % len(rows)]
            m = row["messages"]
            first = [(m[0]["content"], "OK")]
            histories = [
                first+[(m[2]["content"], "OK")],
                first,
                first+[(other["messages"][2]["content"], "OK")],
                [],
            ]
            expected = [
                row["new_code"], row["old_code"],
                other["new_code"], "INCONNU"
            ]
            stats[0] += 1
            for j, (history, answer) in enumerate(
                zip(histories, expected)
            ):
                prefix, _ = app.conversation_prefix(
                    history, m[4]["content"], 512
                )
                result, ended = app.stream_bytes(
                    model, prefix, 96, 0, 40, 42, app.END.encode()
                )
                stats[j+1] += int(
                    ended
                    and result.decode(
                        "utf-8", errors="replace"
                    ).strip() == answer
                )

    for length, stats in sorted(totals.items()):
        n, full, initial, transplanted, unknown = stats
        print(
            f"MISE_A_JOUR L{length} | n={n} | "
            f"complet_nouveau={full}/{n} | "
            f"sans_maj_ancien={initial}/{n} | "
            f"maj_transplantee={transplanted}/{n} | "
            f"sans_histoire_INCONNU={unknown}/{n}",
            flush=True
        )


def smoke():
    rng = random.Random(9)
    row = record(
        "z9q", 0, "mise_a_jour", rng, old_code="r2x"
    )
    assert row["old_code"] != row["new_code"]
    assert len(row["messages"]) == 6

    for item, expected, replies in (
        (row, "OKOKz9q", 3),
        (missing_record(0), "INCONNU", 1),
        (missing_record(0, True), "OKINCONNU", 2),
    ):
        x, y = app.instruction_batch([app.encoded(item)], [0])
        content, boundary, tail = masks3(y)
        assert bytes(y[content].tolist()).decode() == expected
        assert boundary.sum().item() == replies
        assert tail.sum().item() == replies*(len(app.END.encode())-1)
        assert len(app.encoded(item)[0]) <= 512

    extension.smoke()
    print(
        "UPDATE SMOKE OK : trois tours, absence de code "
        "et masques des reponses."
    )
'''

source = Path("assistant_v08.py").read_text(encoding="utf-8-sig")
destination = Path("assistant_v09.py")
if destination.exists():
    raise SystemExit("assistant_v09.py existe deja : aucun remplacement")

patches = [
    (
        'instructions_v08/test.jsonl',
        'instructions_v09/test.jsonl',
        3
    ),
    (
        '"objective": "balanced_copy_extract_dialogue_v1"',
        '"objective": "balanced_update_unknown_v1"',
        1
    ),
    ('def record(', 'def previous_record(', 1),
    (
        '("instructions_v06/test.jsonl", "instructions_v07/fresh_validation.jsonl")',
        '("instructions_v06/test.jsonl", "instructions_v07/fresh_validation.jsonl", "instructions_v08/test.jsonl")',
        2
    ),
    (
        'copie/extraction/dialogue',
        'copie/extraction/rappel/mise_a_jour/inconnu',
        1
    ),
    (
        'runs/assistant_balanced_v07.pt',
        'runs/assistant_dialogue_v08.pt',
        1
    ),
    (
        '            history_controls(args)',
        '            history_controls(args)\n'
        '            update_controls(args)',
        1
    ),
    ('default=800)', 'default=600)', 1),
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
old_batch = 'app.instruction_batch(rows, list(range(4)))'
if tail.count(old_batch) != 1:
    raise SystemExit("Taille du batch inattendue")
tail = tail.replace(
    old_batch,
    'app.instruction_batch(rows, list(range(len(rows))))'
)
new_update = '''def update(model, opt, state, rng, replay_rng, used, reserved):
    step = state["sft"]["step"]
    config = state["sft"]["config"]
    records = [fresh(rng, used, reserved, 8, "copie", 2+step%7)]
    records += [fresh(rng, used, reserved, 5, "extraction", 2+step%4)]
    records += [fresh(rng, used, reserved, 4, "dialogue", 2+step%3)]
    records += [fresh(rng, used, reserved, 4, "mise_a_jour", 2+step%3)]
    records += [missing_record(rng.randrange(4), bool(step%2), rng.randrange(4))]
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
print("assistant_v09.py cree ; syntaxe valide ; anciens fichiers conserves.")
