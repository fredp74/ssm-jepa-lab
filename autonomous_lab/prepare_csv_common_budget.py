import hashlib
import json
from pathlib import Path

from csv_discovery import PROBES

def main():
    output = Path("autonomous_lab/data/csv_common_budget_01")
    if output.exists():
        raise SystemExit("Protocole deja present : aucun remplacement.")

    contrasts = json.loads(Path(
        "autonomous_lab/data/csv_contrasts_01/probe_banks.json"
    ).read_text(encoding="utf-8"))["banks"]["contrasts"]

    base = [["row_B"], ["row_A"]]

    def item(rows, kind="format_consistency", **extra):
        return {"input": rows, "check": kind, **extra}

    def relation(rows, operation):
        return item(
            rows, "decoded_relation",
            baseline_call=1,
            operation=operation,
        )

    plans = {
        "cell_contrasts": [item(probe) for probe in contrasts],
        "structural": [
            item(PROBES[7]),
            item(base),
            relation(list(reversed(base)), "reverse_rows"),
            relation(base + base, "duplicate_rows"),
            relation(base + [[""]], "append_empty_row"),
            relation([[""]] + base, "prepend_empty_row"),
            relation([["row_B", "row_A"]], "transpose"),
        ],
        "combined": [
            item(PROBES[7]),
            item(base),
            relation(list(reversed(base)), "reverse_rows"),
            relation(base + [[""]], "append_empty_row"),
            item([["7", "0007", "1.25", "1,25"]]),
            item([["é_X", "e\u0301_X", "AB_Y", "ＡＢ_Y"]]),
            item([["length_Z"*4]]),
        ],
    }

    metadata = {}
    for name, plan in plans.items():
        calls = len(plan)
        cells = sum(
            len(row) for call in plan for row in call["input"]
        )
        text_bytes = sum(
            len(value.encode("utf-8"))
            for call in plan for row in call["input"] for value in row
        )
        assert calls <= 8 and cells <= 24 and text_bytes <= 512
        encoded = json.dumps(
            plan, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        metadata[name] = {
            "calls": calls,
            "cells": cells,
            "input_text_utf8_bytes": text_bytes,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
        print(f"{name} : {metadata[name]}")

    protocol = {
        "maximum_calls": 8,
        "maximum_cells": 24,
        "maximum_input_text_utf8_bytes": 512,
        "byte_budget_excludes_transport_encoding": True,
        "call_indices_are_zero_based": True,
        "stop_on_detected_contradiction": True,
        "new_behaviors_not_evaluated_yet": True,
        "plans_must_not_be_modified_after_evaluation": True,
        "neural_training": False,
        "interpretation": "comparison_of_handwritten_testing_strategies",
    }

    output.mkdir(parents=True)
    path = output / "plans.json"
    path.write_text(
        json.dumps({
            "protocol": protocol,
            "metadata": metadata,
            "plans": plans,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Plans figes : {path}")

if __name__ == "__main__":
    main()
