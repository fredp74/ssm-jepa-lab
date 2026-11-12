import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch

from csv_environment import configurations, render
from csv_policy_behaviors import simulate
from csv_discovery import DiscoveryAgent
from csv_lookup_policy import LookupPolicy, state_key
from train_csv_ssm_policy import SSMPolicy

torch.set_num_threads(1)

def make_cases(catalog):
    probes = [test["input"] for test in catalog]
    cases = [
        [["e\u0301_hold", " x "]],
        [["2.50", "z"], ["1.25", "a"]],
        [["00031", ""], ["009", ""]],
        [["UPPER_DUP", "yes"], ["UPPER_DUP", "yes"]],
    ]
    rng = random.Random(115001)
    values = [
        "", "00031", "009", "2.50", "-3.75", "YES", "FALSE",
        "e\u0301_hold", " x ", "a  b_hold", "UPPER_hold",
        "comma,hold", 'quote"hold', "line\nhold", "東京_hold",
    ]
    while len(cases) < 100:
        height, width = rng.randint(1, 4), rng.randint(1, 4)
        case = [
            [rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if case not in cases and case not in probes:
            cases.append(case)
    assert not any(case in probes for case in cases)
    return cases

@torch.inference_mode()
def main():
    root = Path("autonomous_lab")
    data = root / "data/csv_ssm_policy_01"
    protocol_raw = (data / "protocol.json").read_bytes()
    protocol = json.loads(protocol_raw.decode("utf-8"))
    catalog = json.loads(
        (data / "catalog.json").read_text(encoding="utf-8")
    )
    output = root / "runs/csv_policy_rollout_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    configs = configurations()
    known_outputs = [
        [render(test["input"], config) for config in configs]
        for test in catalog
    ]

    # Reconstruire uniquement les classes connues de l'enseignant.
    teacher_data = json.loads(
        (root / "runs/csv_symbolic_teacher_01/results.json").read_text(
            encoding="utf-8"
        )
    )
    teacher_responses = []
    for trajectory in teacher_data["trajectories"]:
        alias = trajectory["aliases"][0]
        teacher_responses.append([
            simulate(
                test["input"], configs[alias["configuration"]],
                [alias["operation"]],
            )
            for test in catalog
        ])

    lookup = LookupPolicy(json.loads(
        (data / "lookup_policy.json").read_text(encoding="utf-8")
    )["table"])
    models = {
        f"ssm_{seed}": SSMPolicy.load(
            root / f"runs/csv_ssm_policy_01/model_{seed}.pt"
        )
        for seed in (97001, 97002, 97003)
    }
    policies = ["fixed", "random", "lookup", "symbolic"] + list(models)
    cases = make_cases(catalog)

    environments = [
        ("known", config_id, [])
        for config_id in range(len(configs))
    ] + [
        (f"composition_{index+1}", config_id, operations)
        for index, operations in enumerate(
            protocol["reserved_behavior_compositions"]
        )
        for config_id in range(len(configs))
    ]

    config = {
        "source_protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "policies": policies,
        "known_environments": 48,
        "reserved_composition_environments": 192,
        "reserved_compositions": protocol["reserved_behavior_compositions"],
        "evaluation_updates": 0,
        "random_repetitions": 5,
        "maximum_calls": 8,
        "maximum_cells": 24,
        "maximum_text_bytes": 512,
        "teacher_uses_training_behaviors_only": True,
        "timing_includes_selection_and_known_family_verification": True,
        "timing_excludes_loading_simulator_precomputation_and_final_scoring": True,
        "reserved_compositions_become_consulted_after_this_run": True,
    }
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "evaluation_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results = []
    for environment_id, (group, config_id, operations) in enumerate(environments):
        # References privees de l'evaluateur.
        responses = [
            simulate(test["input"], configs[config_id], operations)
            for test in catalog
        ]
        expected = [
            simulate(case, configs[config_id], operations) for case in cases
        ]
        outside_on_evaluation = not any(
            all(
                render(case, candidate) == answer
                for case, answer in zip(cases, expected)
            )
            for candidate in configs
        )

        for policy in policies:
            repeats = range(5) if policy == "random" else range(1)
            for repetition in repeats:
                rng = random.Random(11501000 + environment_id*10 + repetition)
                if policy in models:
                    models[policy].reset()

                known = list(range(48))
                possible = list(range(len(teacher_responses)))
                used = []
                cells = text_bytes = 0
                unfamiliar = fallbacks = 0

                started = time.perf_counter_ns()
                while known:
                    feasible = [
                        test["test_id"] for test in catalog
                        if test["test_id"] not in used
                        and len(used) < 8
                        and cells + test["cells"] <= 24
                        and text_bytes + test["input_text_utf8_bytes"] <= 512
                    ]
                    if not feasible:
                        break
                    step = {
                        "tested_ids_before": list(used),
                        "known_candidates_before": list(known),
                        "remaining_calls": 8-len(used),
                        "remaining_cells": 24-cells,
                        "remaining_text_bytes": 512-text_bytes,
                        "feasible_actions": feasible,
                    }
                    unfamiliar += state_key(step) not in lookup.table

                    if policy == "fixed":
                        action = min(feasible)
                    elif policy == "random":
                        action = rng.choice(feasible)
                    elif policy == "lookup":
                        action, origin = lookup.choose(step, catalog)
                        fallbacks += origin == "cheapest_feasible_fallback"
                    elif policy in models:
                        action, _ = models[policy].choose(step, catalog)
                    else:
                        def score(test_id):
                            groups = Counter(
                                teacher_responses[i][test_id] for i in possible
                            )
                            known_answers = {
                                known_outputs[test_id][i] for i in known
                            }
                            return (
                                sum(n for answer, n in groups.items()
                                    if answer not in known_answers),
                                len(possible)**2-sum(n*n for n in groups.values()),
                                -catalog[test_id]["cells"],
                                -catalog[test_id]["input_text_utf8_bytes"],
                            )
                        action = max(
                            feasible, key=lambda i: (*score(i), -i)
                        )

                    assert action in feasible
                    answer = responses[action]
                    used.append(action)
                    cells += catalog[action]["cells"]
                    text_bytes += catalog[action]["input_text_utf8_bytes"]
                    known = [
                        i for i in known if known_outputs[action][i] == answer
                    ]
                    if policy == "symbolic":
                        possible = [
                            i for i in possible
                            if teacher_responses[i][action] == answer
                        ]

                elapsed = (time.perf_counter_ns()-started)/1000
                assert len(used) <= 8 and cells <= 24 and text_bytes <= 512

                agent = DiscoveryAgent()
                agent.candidates = known
                predictions = [agent.predict(case) for case in cases]
                exact = sum(
                    prediction is not None and prediction == answer
                    for prediction, answer in zip(predictions, expected)
                )
                results.append({
                    "group": group,
                    "configuration": config_id,
                    "policy": policy,
                    "repetition": repetition,
                    "outside_known_family_on_evaluation": outside_on_evaluation,
                    "detected_unknown": not known,
                    "calls": len(used),
                    "cells": cells,
                    "text_bytes": text_bytes,
                    "unfamiliar_state_decisions": unfamiliar,
                    "lookup_fallbacks": fallbacks,
                    "decision_steps": len(used),
                    "selected_test_ids": used,
                    "rollout_microseconds": elapsed,
                    "answered": sum(p is not None for p in predictions),
                    "exact": exact,
                    "cases": len(cases),
                })

        if (environment_id+1) % 48 == 0:
            print(f"Environnements termines : {environment_id+1}/240", flush=True)

    summaries = []
    for policy in policies:
        for group in ["known"] + [f"composition_{i}" for i in range(1, 5)]:
            rows = [r for r in results if r["policy"] == policy and r["group"] == group]
            summary = {
                "policy": policy, "group": group, "episodes": len(rows),
                "detected_unknown": sum(r["detected_unknown"] for r in rows),
                "false_rejections": sum(
                    r["detected_unknown"]
                    and not r["outside_known_family_on_evaluation"]
                    for r in rows
                ),
                "incorrect_survivors": sum(
                    not r["detected_unknown"] and r["exact"] < r["cases"]
                    for r in rows
                ),
                "fully_correct": sum(r["exact"] == r["cases"] for r in rows),
                "mean_calls": sum(r["calls"] for r in rows)/len(rows),
                "mean_cells": sum(r["cells"] for r in rows)/len(rows),
                "unfamiliar_state_decisions": sum(
                    r["unfamiliar_state_decisions"] for r in rows
                ),
                "decision_steps": sum(r["decision_steps"] for r in rows),
                "mean_rollout_microseconds": sum(
                    r["rollout_microseconds"] for r in rows
                )/len(rows),
            }
            summaries.append(summary)
            print(summary)

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": config, "summaries": summaries, "results": results
        }, indent=2), encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
