import hashlib
import json
import random
from pathlib import Path

from budgeted_five_bits import FAMILY

def signature(rule):
    return "".join(map(str, rule))

def partition(indices):
    key = ",".join(map(str, sorted(indices))).encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return "validation" if value % 5 == 0 else "train"

def main():
    output = Path("autonomous_lab/data/rule_ranking_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    rng = random.Random(91001)
    used_contexts = set()
    contexts = {"train": [], "validation": []}

    # Memes nombres d'observations dans les deux partitions.
    for count in (4, 8, 12):
        for split, target in (("train", 40), ("validation", 10)):
            selected = 0
            while selected < target:
                indices = tuple(sorted(rng.sample(range(32), count)))
                if indices in used_contexts or partition(indices) != split:
                    continue
                used_contexts.add(indices)
                contexts[split].append(indices)
                selected += 1

    output.mkdir(parents=True)
    protocol = {
        "family_size": len(FAMILY),
        "rules": [signature(rule) for rule in FAMILY],
        "observation_counts": [4, 8, 12],
        "contexts_train": 120,
        "contexts_validation": 30,
        "rules_present_in_both_partitions": True,
        "context_partition": "SHA256(sorted_input_indices) modulo 5",
        "training_seed": 91001,
        "task": "rank_known_rules_from_observations",
        "verifier": "exact_consistency_with_every_observation",
        "candidate_compatibility_is_not_proof_of_true_rule": True,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    for split, selected_contexts in contexts.items():
        rows = []
        for indices in selected_contexts:
            for rule_id, rule in enumerate(FAMILY):
                order = list(indices)
                rng.shuffle(order)
                observations = [
                    {
                        "input_index": index,
                        "input": [(index >> shift) & 1 for shift in (4,3,2,1,0)],
                        "output": rule[index],
                    }
                    for index in order
                ]
                compatible = [
                    candidate_id
                    for candidate_id, candidate in enumerate(FAMILY)
                    if all(
                        candidate[item["input_index"]] == item["output"]
                        for item in observations
                    )
                ]
                rows.append({
                    "context": list(indices),
                    "observations": observations,
                    "generating_rule_id": rule_id,
                    "compatible_rule_ids": compatible,
                })

        rng.shuffle(rows)
        path = output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(
            f"{split} : {len(selected_contexts)} contextes, "
            f"{len(rows)} episodes"
        )

    print(f"Donnees : {output}")

if __name__ == "__main__":
    main()
