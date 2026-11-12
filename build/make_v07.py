import ast
from pathlib import Path

functions = r'''
def fresh(rng, used, reserved, maximum, task="copie", length=None):
    length = rng.randint(2, maximum) if length is None else length
    for _ in range(10000):
        payload = "".join(rng.choice(ALPHABET) for _ in range(length))
        if payload not in reserved:
            used.add(payload)
            return record(payload, rng.randrange(4), task, rng, reserved)
    raise RuntimeError("Generation hors test epuisee")


def masks3(y):
    content, ending = app.supervision_masks(y)
    previous = torch.cat([
        torch.zeros_like(ending[:, :1]), ending[:, :-1]
    ], dim=1)
    boundary = ending & ~previous
    return content, boundary, ending & ~boundary


def loss3(logits, y):
    losses = F.cross_entropy(
        logits.reshape(-1, 256), y.reshape(-1), reduction="none"
    ).reshape_as(y)
    masks = masks3(y)
    if not all(mask.any().item() for mask in masks):
        raise ValueError("Groupe de supervision vide")
    return tuple(losses[mask].mean() for mask in masks)


def audit_add(state, row):
    payload = row["messages"][-1]["content"]
    bucket = state["stream_audit"].setdefault(row["task"], [])
    if any(
        item["record"]["messages"][-1]["content"] == payload
        for item in bucket
    ):
        return
    bucket.append({
        "rank": hashlib.sha256(payload.encode()).hexdigest(),
        "record": row
    })
    bucket.sort(key=lambda item: item["rank"])
    del bucket[16:]


def update(model, opt, state, rng, replay_rng, used, reserved):
    step = state["sft"]["step"]
    config = state["sft"]["config"]
    records = [
        fresh(rng, used, reserved, 8, "copie", 2+(2*step+j)%7)
        for j in range(2)
    ]
    records += [
        fresh(rng, used, reserved, 5, "extraction", 2+(2*step+j)%4)
        for j in range(2)
    ]
    rows = [app.encoded(row) for row in records]
    x, y = app.instruction_batch(rows, list(range(4)))
    content, boundary, tail = loss3(model(x), y)

    indices = torch.randint(
        len(state["memory"]), (2,), generator=replay_rng
    )
    replay = state["memory"][indices].long()
    lm = F.cross_entropy(
        model(replay[:, :-1]).reshape(-1, 256),
        replay[:, 1:].reshape(-1)
    )
    instruction = 0.75*content+0.20*boundary+0.05*tail
    loss = (
        (1-config["replay_weight"])*instruction
        + config["replay_weight"]*lm
    )
    if not torch.isfinite(loss).item():
        raise RuntimeError("Perte non finie")

    lr = config["lr"]*min(1.0, (step+1)/20)*(
        0.1+0.45*(1+math.cos(math.pi*step/config["steps"]))
    )
    for group in opt.param_groups:
        group["lr"] = lr
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    opt.step()

    for row in records:
        audit_add(state, row)
        key = row["task"]
        state["balanced_counts"][key] = (
            state["balanced_counts"].get(key, 0)+1
        )
    state["sft"]["step"] += 1
    return 8, [
        value.item()/math.log(2)
        for value in (content, boundary, tail, lm)
    ]


@torch.no_grad()
def boundary_report(args):
    model = extension.factory(core.load(args.run)).eval()
    records, _ = answers(args.data)
    rows = app.examples(args.data)
    for task in sorted({row["task"] for row in records}):
        indices = [
            i for i, row in enumerate(records) if row["task"] == task
        ]
        total = count = correct = 0
        for start in range(0, len(indices), 4):
            x, y = app.instruction_batch(rows, indices[start:start+4])
            logits = model(x)
            losses = F.cross_entropy(
                logits.reshape(-1, 256),
                y.reshape(-1), reduction="none"
            ).reshape_as(y)
            boundary = masks3(y)[1]
            total += losses[boundary].sum().item()
            count += boundary.sum().item()
            correct += (
                logits.argmax(-1)[boundary] == y[boundary]
            ).sum().item()
        print(
            f"DEBUT_FIN {task} | {total/count/math.log(2):.4f} bpb | "
            f"premier octet correct {correct}/{count} "
            "avec prefixe de reference",
            flush=True
        )


def smoke():
    rng = random.Random(7)
    used, reserved = {"aa"}, {"bb", "12"}
    row = fresh(rng, used, reserved, 5, "extraction", 2)
    assert len(row["messages"][-1]["content"]) == 2
    assert not set(row["source_values"]) & reserved

    rows = [
        app.encoded(record("xy", 0)),
        app.encoded(record("abcd", 1))
    ]
    x, y = app.instruction_batch(rows, [0, 1])
    content, boundary, tail = masks3(y)
    assert content.sum().item() == 6
    assert boundary.sum().item() == 2
    assert tail.sum().item() == 2*(len(app.END.encode())-1)
    assert (y[boundary] == app.END.encode()[0]).all()

    for period, maximum in ((7, 8), (4, 5)):
        counts = {n: 0 for n in range(2, maximum+1)}
        for step in range(period):
            for j in range(2):
                counts[2+(2*step+j)%period] += 1
        assert set(counts.values()) == {2}

    extension.smoke()
    print(
        "BALANCED SMOKE OK : longueurs equilibrees et masques "
        "contenu/debut-fin/suite-fin."
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")

    p = sub.add_parser("train")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--test", type=Path,
        default=Path("instructions_v06/test.jsonl")
    )
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--replay-weight", type=float, default=0.3)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--data", type=Path,
        default=Path("instructions_v06/test.jsonl")
    )
    p.add_argument("--generate", action="store_true")
    p.add_argument("--limit-per-task", type=int, default=16)

    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "eval":
        boundary_report(args)
        app.evaluate(args)
    else:
        if (
            args.steps < 4 or args.lr <= 0
            or not 0 <= args.replay_weight < 1
        ):
            parser.error("Parametres invalides")
        train(args)
        state = core.load(args.run)
        print(
            "EXEMPLES DE CETTE ETAPE : "
            + json.dumps(state["balanced_counts"], sort_keys=True)
        )
'''

source = Path("assistant_v06.py").read_text(encoding="utf-8-sig")
destination = Path("assistant_v07.py")
if destination.exists():
    raise SystemExit("assistant_v07.py existe deja : aucun remplacement")

patches = [
    (
        '"objective": "fresh_copy_extraction_v1"',
        '"objective": "balanced_copy_extraction_boundary_v1"'
    ),
    (
        'state["stream_audit"] = {}',
        'state["stream_audit"] = {}\n'
        '        state["balanced_counts"] = {}'
    ),
    (
        'fin {losses[1]:.3f}',
        'debut_fin {losses[1]:.3f} | suite_fin {losses[2]:.3f}'
    ),
    ('replay {losses[2]:.3f}', 'replay {losses[3]:.3f}'),
    ('nouvelles cibles', 'cibles distinctes supplementaires'),
]
for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

to_add = []
for replacement in ast.parse(functions).body:
    if not isinstance(replacement, ast.FunctionDef):
        continue
    tree = ast.parse(source)
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == replacement.name
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
print("assistant_v07.py cree ; syntaxe valide ; anciens fichiers conserves.")
