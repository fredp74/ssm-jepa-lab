import argparse
import json
from pathlib import Path

import torch
import assistant_v09 as trainer

app = trainer.app

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    model = trainer.extension.factory(trainer.core.load(args.run)).eval()
    records = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    failures = 0
    print(f"MODELE : {args.run}", flush=True)

    for number, record in enumerate(records, 1):
        history = []
        turns = []
        success = True
        messages = record["messages"]

        for position in range(0, len(messages), 2):
            question = messages[position]["content"]
            expected = messages[position + 1]["content"]
            prefix, history = app.conversation_prefix(history, question, 512)
            result, ended = app.stream_bytes(
                model, prefix, 96, 0, 40, 42, app.END.encode()
            )
            answer = result.decode("utf-8", errors="replace")
            correct = answer.strip() == expected.strip() and ended
            success = success and correct
            turns.append((question, expected, answer, ended, correct))
            history.append((question, answer))

        if success:
            continue

        failures += 1
        print(f"\nECHEC #{number} | {record['task']}", flush=True)
        for turn, (question, expected, answer, ended, correct) in enumerate(turns, 1):
            print(f"  Tour {turn} | correct={correct}")
            print(f"  Question : {question!r}")
            print(f"  Attendu  : {expected!r}")
            print(f"  Obtenu   : {answer!r} | EOS={ended}", flush=True)

    print(
        f"\nBILAN : {len(records)-failures}/{len(records)} reussis"
        f" | {failures} echecs",
        flush=True,
    )

if __name__ == "__main__":
    main()
