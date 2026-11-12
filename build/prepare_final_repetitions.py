import json
import random
import string
from pathlib import Path

import assistant_v09 as trainer

output = Path("final_evaluation/repetitions.jsonl")
if output.exists():
    raise SystemExit("Test deja present : aucun remplacement.")

alphabet = string.ascii_lowercase + string.digits
reserved = set()

for checkpoint in (
    "runs/assistant_update_v09.pt",
    "runs/assistant_longcopy_v10.pt",
    "runs/experiment_v13_control.pt",
    "runs/experiment_v13_repetition.pt",
    "runs/experiment_v13_control_130014.pt",
    "runs/experiment_v13_repetition_130014.pt",
    "runs/experiment_v13_control_130015.pt",
    "runs/experiment_v13_repetition_130015.pt",
):
    state = trainer.core.load(Path(checkpoint))
    reserved.update(state.get("stream_used", []))

for path in Path(".").glob("instructions_*/*.jsonl"):
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            reserved.update(row.get("source_values", []))
            for message in row.get("messages", []):
                value = message.get("content", "")
                if (
                    message.get("role") == "assistant"
                    and 2 <= len(value) <= 8
                    and all(c in alphabet for c in value)
                ):
                    reserved.add(value)

rng = random.Random(26011202)
rows = []

for length in (6, 7, 8):
    for index in range(32):
        for attempt in range(100000):
            base = rng.sample(alphabet, length)

            adjacent = base.copy()
            position = rng.randrange(length - 1)
            adjacent[position + 1] = adjacent[position]

            spaced = base.copy()
            first, last = sorted(rng.sample(range(length), 2))
            if last - first < 2:
                continue
            spaced[last] = spaced[first]

            values = [
                "".join(base),
                "".join(adjacent),
                "".join(spaced),
            ]
            if all(value not in reserved for value in values):
                break
        else:
            raise RuntimeError("Generation epuisee.")

        for group, value in zip(
            ("sans_repetition", "consecutive", "espacee"), values
        ):
            reserved.add(value)
            row = trainer.record(value, index % 4, "copie")
            row["task"] = f"copie_L{length}_{group}"
            row["pair_id"] = f"L{length}_{index:02d}"
            rows.append(row)

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
print(f"Test cree : {output} | {len(rows)} cas")
print("32 cas par longueur et type de repetition.")
