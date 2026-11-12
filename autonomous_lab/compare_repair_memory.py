import json
from collections import defaultdict
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    output = Path("autonomous_lab/runs/repair_memory_control_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    initial = read(
        "autonomous_lab/runs/budgeted_five_bits_01/results.json"
    )
    repairs = read(
        "autonomous_lab/runs/local_repair_01/results.json"
    )
    cases = [
        row for row in initial["details"]
        if row["status"] == "provisional"
    ]
    if len(cases) != len(repairs["details"]):
        raise ValueError("Nombre de cas incompatible.")

    summaries = defaultdict(lambda: {
        "cases": 0,
        "unqueried_inputs": 0,
        "original_correct": 0,
        "memory_correct": 0,
        "memory_exact_functions": 0,
        "accepted_repair_cases": 0,
        "repair_evaluated_inputs": 0,
        "repair_correct": 0,
        "repair_exact_functions": 0,
    })
    details = []

    for case_id, (old, repaired) in enumerate(
        zip(cases, repairs["details"])
    ):
        assert old["group"] == repaired["group"]
        assert old["proposal"] == repaired["original_proposal"]
        hidden = old["hidden_table"]
        original = old["proposal"]
        observed = {
            item["input_index"]: str(item["answer"])
            for item in repaired["observations"]
        }
        assert all(hidden[i] == answer for i, answer in observed.items())

        patched = list(original)
        for index, answer in observed.items():
            patched[index] = answer
        patched = "".join(patched)

        remaining = [i for i in range(32) if i not in observed]
        original_score = sum(original[i] == hidden[i] for i in remaining)
        memory_score = sum(patched[i] == hidden[i] for i in remaining)
        assert original_score == memory_score

        key = f"{old['group']} / {repaired['status']}"
        summary = summaries[key]
        summary["cases"] += 1
        summary["unqueried_inputs"] += len(remaining)
        summary["original_correct"] += original_score
        summary["memory_correct"] += memory_score
        summary["memory_exact_functions"] += patched == hidden

        accepted = repaired["status"] in (
            "unchanged_provisional", "repair_provisional"
        )
        repair_score = None
        if accepted:
            final = repaired["final_proposal"]
            repair_score = sum(final[i] == hidden[i] for i in remaining)
            summary["accepted_repair_cases"] += 1
            summary["repair_evaluated_inputs"] += len(remaining)
            summary["repair_correct"] += repair_score
            summary["repair_exact_functions"] += final == hidden

        details.append({
            "case_id": case_id,
            "group": old["group"],
            "status": repaired["status"],
            "unqueried_inputs": len(remaining),
            "original_correct": original_score,
            "memory_correct": memory_score,
            "accepted_repair_correct": repair_score,
            "memory_exact": patched == hidden,
        })

    for key, summary in summaries.items():
        print(f"\n{key}")
        print(json.dumps(summary, ensure_ascii=False))

    output.mkdir(parents=True)
    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "new_environment_queries": 0,
                "control": "original_rule_overridden_on_all_observed_inputs",
                "evaluation": "same_unqueried_inputs_for_each_case",
                "analysis": "retrospective",
                "rejected_repairs_are_not_used_for_prediction": True,
            },
            "summaries": dict(summaries),
            "details": details,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
