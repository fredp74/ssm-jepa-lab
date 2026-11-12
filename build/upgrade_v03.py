import ast
from pathlib import Path

extra = r'''
def supervision_masks(y):
    marker = torch.tensor(list(END.encode()), dtype=torch.long)
    ending = torch.zeros_like(y, dtype=torch.bool)
    if y.size(1) >= len(marker):
        matches = y.unfold(1, len(marker), 1).eq(marker).all(-1)
        for row, start in matches.nonzero().tolist():
            ending[row, start:start+len(marker)] = True
    return (y != -100) & ~ending, ending


def split_instruction_loss(logits, y):
    losses = F.cross_entropy(
        logits.reshape(-1, 256), y.reshape(-1), reduction="none"
    ).reshape_as(y)
    content, ending = supervision_masks(y)
    if not content.any() or not ending.any():
        raise ValueError("Contenu ou marqueur de fin absent du batch")
    return losses[content].mean(), losses[ending].mean()


@torch.no_grad()
def evaluate(args):
    if args.limit_per_task < 1:
        raise ValueError("limit-per-task doit etre positif")
    model = gru_model(core.load(args.run)).eval()
    records = [
        json.loads(line)
        for line in args.data.read_text("utf-8-sig").splitlines()
        if line.strip()
    ]
    rows = examples(args.data)
    tasks = sorted({r.get("task", "autre") for r in records})

    for task in tasks:
        indices = [
            i for i, r in enumerate(records)
            if r.get("task", "autre") == task
        ]
        totals, counts = [0.0, 0.0], [0, 0]

        for start in range(0, len(indices), 4):
            x, y = instruction_batch(rows, indices[start:start+4])
            losses = F.cross_entropy(
                model(x).reshape(-1, 256),
                y.reshape(-1), reduction="none"
            ).reshape_as(y)
            for j, mask in enumerate(supervision_masks(y)):
                totals[j] += losses[mask].sum().item()
                counts[j] += mask.sum().item()

        bits = [
            totals[j]/max(counts[j], 1)/math.log(2)
            for j in range(2)
        ]
        print(
            f"PERTE {task} | n={len(indices)} | "
            f"contenu={bits[0]:.4f} bpb ({counts[0]} octets) | "
            f"fin={bits[1]:.4f} bpb ({counts[1]} octets)",
            flush=True
        )

        if not args.generate:
            continue

        gen = torch.Generator().manual_seed(42)
        order = torch.randperm(
            len(indices), generator=gen
        )[:args.limit_per_task].tolist()

        content_ok = eos_ok = joint_ok = dialogue_ok = 0
        for number, selected in enumerate(order):
            record = records[indices[selected]]
            history = []
            entire = True

            for i in range(0, len(record["messages"]), 2):
                question = record["messages"][i]["content"]
                expected = record["messages"][i+1]["content"]
                prefix, history = conversation_prefix(
                    history, question, 512
                )
                result, ended = stream_bytes(
                    model, prefix, 96, 0, 40, 42, END.encode()
                )
                answer = result.decode("utf-8", errors="replace")
                correct = answer.strip() == expected.strip()
                entire = entire and correct and ended
                history.append((question, answer))

            content_ok += int(correct)
            eos_ok += int(ended)
            joint_ok += int(correct and ended)
            dialogue_ok += int(entire)

            if number < 2:
                print(
                    f"EXEMPLE {task} | attendu={expected!r} | "
                    f"obtenu={answer!r} | EOS={ended}",
                    flush=True
                )

        n = len(order)
        print(
            f"GENERATIONS {task} | n={n} | "
            f"contenu_final={content_ok}/{n} | "
            f"EOS_final={eos_ok}/{n} | "
            f"exact_final+EOS={joint_ok}/{n} | "
            f"tous_les_tours_exacts+EOS={dialogue_ok}/{n}",
            flush=True
        )
'''

original = Path("assistant_v02.py")
destination = Path("assistant_v03.py")

if not original.exists():
    raise SystemExit("assistant_v02.py absent : mise a niveau impossible.")
if destination.exists():
    raise SystemExit("assistant_v03.py existe deja : aucun remplacement.")

source = original.read_text(encoding="utf-8-sig")

patches = [
    (
        '"format": 1,',
        '"format": 2, "instruction_loss": "0.9content+0.1end",'
    ),
    (
        '            instruction_loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))',
        '            content_loss, end_loss = split_instruction_loss(model(x), y)\n'
        '            instruction_loss = 0.9*content_loss + 0.1*end_loss'
    ),
    (
        'reponse {instruction_loss.item()/math.log(2):.3f} bpb',
        'contenu {content_loss.item()/math.log(2):.3f} bpb | fin {end_loss.item()/math.log(2):.3f} bpb'
    ),
    (
        '            p.add_argument("--generate", action="store_true")',
        '            p.add_argument("--generate", action="store_true")\n'
        '            p.add_argument("--limit-per-task", type=int, default=16)'
    ),
]

for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

tree = ast.parse(source)
functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
]
if len(functions) != 1:
    raise SystemExit("Fonction evaluate introuvable ou multiple.")

node = functions[0]
first = min(
    [node.lineno] + [d.lineno for d in node.decorator_list]
)-1
lines = source.splitlines(keepends=True)
source = (
    "".join(lines[:first]) + extra + "\n"
    + "".join(lines[node.end_lineno:])
)

compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print("assistant_v03.py cree ; syntaxe valide ; v0.2 conservee.")
