import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from csv_environment import configurations, render
from csv_policy_behaviors import simulate

def main():
    source = Path("autonomous_lab/data/csv_ssm_policy_01")
    protocol = json.loads(
        (source / "protocol.json").read_text(encoding="utf-8")
    )
    catalog = json.loads(
        (source / "catalog.json").read_text(encoding="utf-8")
    )
    output = Path("autonomous_lab/runs/csv_branched_teacher_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    configs = configurations()
    operations = ["identity"] + protocol[
        "training_behaviors"
    ]["individual_operations"]

    known_outputs = [
        [render(test["input"], config) for config in configs]
        for test in catalog
    ]

    # Un seul poids par comportement observable sur le catalogue.
    classes = []
    by_signature = {}
    for operation in operations:
        for config_id, config in enumerate(configs):
            responses = [
                simulate(test["input"], config, [operation])
                for test in catalog
            ]
            signature = hashlib.sha256(
                json.dumps(responses, ensure_ascii=False).encode("utf-8")
            ).hexdigest()

            if signature in by_signature:
                classes[by_signature[signature]]["aliases"].append({
                    "operation": operation,
                    "configuration": config_id,
                })
            else:
                by_signature[signature] = len(classes)
                classes.append({
                    "responses": responses,
                    "signature": signature,
                    "aliases": [{
                        "operation": operation,
                        "configuration": config_id,
                    }],
                })

    def feasible(test_id, used, cells, text_bytes):
        test = catalog[test_id]
        return (
            test_id not in used
            and len(used) < protocol["maximum_calls"]
            and cells + test["cells"] <= protocol["maximum_cells"]
            and text_bytes + test["input_text_utf8_bytes"]
                <= protocol["maximum_input_text_utf8_bytes"]
        )

    trajectories = []
    for rollout_id in range(len(classes)*4):
        environment_id = rollout_id // 4
        repetition = rollout_id % 4
        environment = classes[environment_id]
        rng = random.Random(98001000 + rollout_id)
        possible = list(range(len(classes)))
        known = list(range(len(configs)))
        used = []
        cells = text_bytes = 0
        steps = []

        while known:
            available = [
                test["test_id"] for test in catalog
                if feasible(test["test_id"], used, cells, text_bytes)
            ]
            if not available:
                break

            scores = {}
            for test_id in available:
                response_groups = Counter(
                    classes[index]["responses"][test_id]
                    for index in possible
                )
                known_responses = {
                    known_outputs[test_id][index] for index in known
                }
                detection_count = sum(
                    count for response, count in response_groups.items()
                    if response not in known_responses
                )
                information_score = (
                    len(possible)**2
                    - sum(count*count for count in response_groups.values())
                )
                test = catalog[test_id]
                scores[test_id] = (
                    detection_count,
                    information_score,
                    -test["cells"],
                    -test["input_text_utf8_bytes"],
                )

            best = max(scores.values())
            best_actions = sorted(
                index for index, score in scores.items() if score == best
            )
            selected = rng.choice(best_actions)
            response = environment["responses"][selected]

            # Etat observable avant l'action : aucune identite cachee.
            step = {
                "tested_ids_before": list(used),
                "known_candidates_before": list(known),
                "remaining_calls": protocol["maximum_calls"]-len(used),
                "remaining_cells": protocol["maximum_cells"]-cells,
                "remaining_text_bytes": (
                    protocol["maximum_input_text_utf8_bytes"]-text_bytes
                ),
                "feasible_actions": available,
                "teacher_best_actions": best_actions,
                "selected_test_id": selected,
                "response": response,
            }

            used.append(selected)
            cells += catalog[selected]["cells"]
            text_bytes += catalog[selected]["input_text_utf8_bytes"]
            possible = [
                index for index in possible
                if classes[index]["responses"][selected] == response
            ]
            known = [
                index for index in known
                if known_outputs[selected][index] == response
            ]
            assert environment_id in possible
            step["known_candidates_after"] = list(known)
            steps.append(step)

        observable_known = any(
            all(
                environment["responses"][i] == known_outputs[i][candidate]
                for i in range(len(catalog))
            )
            for candidate in range(len(configs))
        )

        trajectories.append({
            "environment_class_id": environment_id,
            "tie_breaking_repetition": repetition,
            "observable_signature": environment["signature"],
            "aliases": environment["aliases"],
            "matches_known_on_entire_catalog": observable_known,
            "detected_unknown": not known,
            "calls": len(used),
            "cells": cells,
            "text_bytes": text_bytes,
            "steps": steps,
        })

    summary = {
        "observable_classes": len(classes),
        "trajectories": len(trajectories),
        "decision_steps": sum(len(row["steps"]) for row in trajectories),
        "distinct_decision_states": len({
            (tuple(step["tested_ids_before"]),
             tuple(step["known_candidates_before"]),
             step["remaining_calls"], step["remaining_cells"],
             step["remaining_text_bytes"])
            for row in trajectories for step in row["steps"]
        }),
        "known_trajectories": sum(
            row["matches_known_on_entire_catalog"] for row in trajectories
        ),
        "detectable_trajectories": sum(
            not row["matches_known_on_entire_catalog"] for row in trajectories
        ),
        "detected_unknown_trajectories": sum(
            row["detected_unknown"] for row in trajectories
        ),
        "false_rejections": sum(
            row["detected_unknown"] and row["matches_known_on_entire_catalog"]
            for row in trajectories
        ),
        "mean_calls": sum(row["calls"] for row in trajectories)/len(trajectories),
        "reserved_compositions_evaluated": False,
    }

    output.mkdir(parents=True)
    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "teacher": "greedy_detection_then_information_then_cost",
                "globally_optimal_policy": False,
                "trajectories_per_observable_class": 4,
                "tie_breaking": "uniform_random_among_best_actions",
                "tie_breaking_seed_base": 98001000,
                "student_induced_states_collected": False,
                "prior": "uniform_over_distinct_catalog_response_classes",
                "catalog_size": len(catalog),
                "maximum_calls": protocol["maximum_calls"],
                "maximum_cells": protocol["maximum_cells"],
                "maximum_text_bytes": protocol["maximum_input_text_utf8_bytes"],
                "training_operations": operations,
                "reserved_compositions_used": False,
                "student_observation_features": [
                    "tested_ids",
                    "known_candidate_mask",
                    "remaining_budgets",
                ],
            },
            "summary": summary,
            "trajectories": trajectories,
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Trajectoires : {path}")

if __name__ == "__main__":
    main()
