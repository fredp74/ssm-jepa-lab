import hashlib
import json
from collections import Counter
from pathlib import Path

def state_key(step):
    return (
        tuple(step["tested_ids_before"]),
        tuple(step["known_candidates_before"]),
        step["remaining_calls"],
        step["remaining_cells"],
        step["remaining_text_bytes"],
    )

def main():
    source = Path("autonomous_lab/runs/csv_branched_teacher_01/results.json")
    raw = source.read_bytes()
    teacher = json.loads(raw.decode("utf-8"))
    output = Path("autonomous_lab/runs/csv_policy_dataset_02")
    manifest_path = Path(
        "autonomous_lab/data/csv_ssm_policy_01/trajectory_split_02.json"
    )
    if output.exists() or manifest_path.exists():
        raise SystemExit("Decoupage deja present : aucun remplacement.")

    split = {"train": [], "validation": []}
    targets = {}
    conflicts = []

    for trajectory in teacher["trajectories"]:
        signature = trajectory["observable_signature"]
        partition = (
            "validation"
            if int(signature[:8], 16) % 5 == 0
            else "train"
        )
        split[partition].append(trajectory)

        for step in trajectory["steps"]:
            key = state_key(step)
            target = tuple(sorted(step["teacher_best_actions"]))
            if key in targets and targets[key] != target:
                conflicts.append({
                    "environment": trajectory["environment_class_id"],
                    "first_target": targets[key],
                    "other_target": target,
                })
            targets[key] = target

    if conflicts:
        raise SystemExit(
            f"{len(conflicts)} conflits de cibles pour un meme etat : "
            "ne pas entrainer avant de revoir les caracteristiques."
        )

    train_signatures = {
        row["observable_signature"] for row in split["train"]
    }
    validation_signatures = {
        row["observable_signature"] for row in split["validation"]
    }
    assert not train_signatures & validation_signatures

    train_states = {
        state_key(step)
        for trajectory in split["train"]
        for step in trajectory["steps"]
    }
    counts = Counter()

    for trajectory in split["validation"]:
        for step in trajectory["steps"]:
            shared = state_key(step) in train_states
            counts["shared_state_decisions" if shared else "unseen_state_decisions"] += 1

    summary = {
        "train_trajectories": len(split["train"]),
        "validation_trajectories": len(split["validation"]),
        "train_decisions": sum(len(t["steps"]) for t in split["train"]),
        "validation_decisions": sum(
            len(t["steps"]) for t in split["validation"]
        ),
        "distinct_train_states": len(train_states),
        "validation_shared_state_decisions": counts["shared_state_decisions"],
        "validation_unseen_state_decisions": counts["unseen_state_decisions"],
        "target_conflicts": len(conflicts),
        "reserved_compositions_used": False,
    }

    manifest = {
        "teacher_source_sha256": hashlib.sha256(raw).hexdigest(),
        "partition": "first_32_bits_of_observable_signature_modulo_5",
        "validation_remainder": 0,
        "trajectory_ids": {
            name: [t["environment_class_id"] for t in trajectories]
            for name, trajectories in split.items()
        },
        "summary": summary,
    }

    output.mkdir(parents=True)
    for name, trajectories in split.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for trajectory in trajectories:
                # Ajouter l'indicateur pour les futures mesures d'imitation.
                copy = dict(trajectory)
                copy["steps"] = [
                    {
                        **step,
                        "state_seen_in_train": state_key(step) in train_states,
                    }
                    for step in trajectory["steps"]
                ]
                stream.write(json.dumps(copy, ensure_ascii=False) + "\n")

    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Manifeste : {manifest_path}")
    print(f"Donnees locales : {output}")

if __name__ == "__main__":
    main()
