import json
import random
from pathlib import Path
import assistant_v09 as app

output = Path("final_evaluation/general.jsonl")
if output.exists():
    raise SystemExit("Validation deja presente : aucun remplacement.")

reserved = set()

# Exclure les cibles d'entrainement enregistrees des deux modeles.
for name in (
    "runs/assistant_update_v09.pt",
    "runs/assistant_longcopy_v10.pt",
    "runs/experiment_v13_control.pt",
    "runs/experiment_v13_repetition.pt",
    "runs/experiment_v13_control_130014.pt",
    "runs/experiment_v13_repetition_130014.pt",
    "runs/experiment_v13_control_130015.pt",
    "runs/experiment_v13_repetition_130015.pt",
):
    state = app.core.load(Path(name))
    reserved.update(state.get("stream_used", []))

# Exclure aussi les valeurs des anciens jeux et audits disponibles.
paths = list(Path(".").glob("instructions_*/*.jsonl"))
paths += list(Path("runs").glob("*.train_audit.jsonl"))
for path in paths:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reserved.update(row.get("source_values", []))
        for message in row.get("messages", []):
            value = message.get("content", "")
            if (
                message.get("role") == "assistant"
                and 2 <= len(value) <= 8
                and all(c in app.ALPHABET for c in value)
            ):
                reserved.add(value)

rng = random.Random(26011201)
rows = []

def new_code(length):
    for _ in range(100000):
        value = "".join(rng.choice(app.ALPHABET) for _ in range(length))
        if value not in reserved:
            reserved.add(value)
            return value
    raise RuntimeError(f"Codes disponibles epuises pour longueur {length}")

for task, maximum in (
    ("copie", 8),
    ("extraction", 5),
    ("dialogue", 4),
    ("mise_a_jour", 4),
):
    for length in range(3, maximum + 1):
        for index in range(16):
            payload = new_code(length)
            options = {}
            if task in ("dialogue", "mise_a_jour"):
                options["last_style"] = index // 4
            if task == "mise_a_jour":
                options["old_code"] = new_code(length)
                options["update_style"] = (index + index // 4) % 4
            row = app.record(
                payload, index % 4, task, rng, reserved, **options
            )
            reserved.update(row.get("source_values", []))
            rows.append(row)

rows += [app.missing_record(style) for style in range(4)]
rows += [
    app.missing_record(style, True, neutral)
    for neutral in range(4)
    for style in range(4)
]

for row in rows:
    app.app.encoded(row)

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
print(f"Validation creee : {output} | {len(rows)} cas")
print("Nouveaux codes ; formulations habituelles conservees.")
