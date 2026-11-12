import hashlib
import json
import statistics
import time
from pathlib import Path

def state_key(step):
    return json.dumps([
        step["tested_ids_before"],
        step["known_candidates_before"],
        step["remaining_calls"],
        step["remaining_cells"],
        step["remaining_text_bytes"],
    ], separators=(",", ":"))

class LookupPolicy:
    def __init__(self, table):
        self.table = table

    def choose(self, step, catalog):
        feasible = step["feasible_actions"]
        if not feasible:
            return None, "no_feasible_action"

        remembered = self.table.get(state_key(step), [])
        available = [action for action in remembered if action in feasible]
        if available:
            return min(available), "lookup"

        action = min(
            feasible,
            key=lambda i: (
                catalog[i]["cells"],
                catalog[i]["input_text_utf8_bytes"],
                i,
            ),
        )
        return action, "cheapest_feasible_fallback"

def read_trajectories(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def main():
    root = Path("autonomous_lab")
    data = root / "runs/csv_policy_dataset_01"
    output = root / "runs/csv_lookup_baseline_01"
    policy_path = root / "data/csv_ssm_policy_01/lookup_policy.json"
    if output.exists() or policy_path.exists():
        raise SystemExit("Reference deja presente : aucun remplacement.")

    catalog = json.loads(
        (root / "data/csv_ssm_policy_01/catalog.json").read_text(
            encoding="utf-8"
        )
    )
    train_path = data / "train.jsonl"
    validation_path = data / "validation.jsonl"
    train = read_trajectories(train_path)
    validation = read_trajectories(validation_path)

    table = {}
    for trajectory in train:
        for step in trajectory["steps"]:
            key = state_key(step)
            actions = sorted(step["teacher_best_actions"])
            if key in table and table[key] != actions:
                raise ValueError("Cibles contradictoires pour un meme etat.")
            table[key] = actions

    policy = LookupPolicy(table)
    steps = [
        step for trajectory in validation for step in trajectory["steps"]
    ]
    matches = hits = fallbacks = 0

    for step in steps:
        action, origin = policy.choose(step, catalog)
        assert action in step["feasible_actions"]
        matches += action in step["teacher_best_actions"]
        hits += origin == "lookup"
        fallbacks += origin == "cheapest_feasible_fallback"

    # Controle artificiel du repli, sans comportement reserve.
    unknown_state = dict(steps[0])
    unknown_state["tested_ids_before"] = [-1]
    fallback_action, origin = policy.choose(unknown_state, catalog)
    assert origin == "cheapest_feasible_fallback"
    assert fallback_action in unknown_state["feasible_actions"]

    for step in steps:
        policy.choose(step, catalog)

    timings = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(100):
            for step in steps:
                policy.choose(step, catalog)
        elapsed = time.perf_counter_ns()-start
        timings.append(elapsed/(100*len(steps))/1000)

    config = {
        "policy": "exact_state_lookup",
        "fallback": "minimum_cells_then_bytes_then_id",
        "training_source_sha256": hashlib.sha256(
            train_path.read_bytes()
        ).hexdigest(),
        "validation_source_sha256": hashlib.sha256(
            validation_path.read_bytes()
        ).hexdigest(),
        "reserved_compositions_used": False,
        "state_key_serialization_included_in_timing": True,
        "file_loading_excluded_from_timing": True,
    }
    summary = {
        "memorized_states": len(table),
        "validation_decisions": len(steps),
        "teacher_action_matches": matches,
        "lookup_hits": hits,
        "fallback_calls": fallbacks,
        "synthetic_fallback_check_passed": True,
        "median_microseconds_per_decision": statistics.median(timings),
        "microseconds_per_decision_repetitions": timings,
        "unseen_state_generalization_measured": False,
    }

    policy_path.write_text(
        json.dumps({
            "config": config,
            "table": table,
        }, indent=2),
        encoding="utf-8",
    )
    restored = LookupPolicy(
        json.loads(policy_path.read_text(encoding="utf-8"))["table"]
    )
    assert all(
        restored.choose(step, catalog) == policy.choose(step, catalog)
        for step in steps
    )

    output.mkdir(parents=True)
    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": config,
            "summary": summary,
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Politique sauvegardee : {policy_path}")
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
