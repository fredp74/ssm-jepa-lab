import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from csv_environment import configurations
from csv_policy_behaviors import simulate


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    root = Path("autonomous_lab")
    transfer_data = root / "data/csv_transfer_01"
    transfer_protocol = read(transfer_data / "protocol.json")
    fingerprints = dict(
        transfer_protocol["agent_and_dependencies_sha256"]
    )
    fingerprints.update({
        str(transfer_data / name): digest
        for name, digest in
        transfer_protocol["prepared_files_sha256"].items()
    })
    for name, expected in fingerprints.items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Empreinte modifiee : {name}")
    protocol = read(root / "data/csv_composition_01/protocol.json")
    protocol["environments"] = read(
        transfer_data / "environments.json"
    )
    catalog = read(root / "data/csv_ssm_policy_01/catalog.json")
    output = root / "runs/csv_transfer_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    configs = configurations()
    programs = [
        {"configuration_id": configuration_id, "operations": sequence}
        for configuration_id in range(len(configs))
        for sequence in protocol["candidate_operation_sequences"]
    ]

    def execute(program, rows):
        if program.get("extra_operation_after_sequence"):
            if (
                program["extra_operation_after_sequence"]
                != "reverse_columns"
                or program["operations"]
            ):
                raise RuntimeError("Operation externe inattendue.")
            rows = [list(reversed(row)) for row in rows]
        return simulate(
            rows,
            configs[program["configuration_id"]],
            program["operations"],
        )

    # Deux appels, quatre cellules chacun, moins de 128 octets.
    learning_checks = [
        [["A", "00042"], ["a", "00042"]],
        [["  ", ""], ["e\u0301_verify ", "7.25"]],
    ]
    checks = [
        [
            ["z_verify"],
            ["a_verify"],
            ["abcdefghijklmnop_X"],
            ["abcdefghijklmnop_Y"],
        ],
        [
            ["e\u0301_check "],
            ["é_check "],
            ["   "],
            ["00048"],
        ],
    ]
    check_cells = sum(len(row) for rows in checks for row in rows)
    check_bytes = sum(
        len(value.encode("utf-8"))
        for rows in checks for row in rows for value in row
    )
    if check_cells > 8 or check_bytes > 128:
        raise RuntimeError("Budget de verification depasse.")
    if any(rows == t["input"] for rows in checks for t in catalog):
        raise RuntimeError("Verification presente dans le catalogue.")

    output.mkdir(parents=True)
    write(output / "protocol.json", {
        "source_protocol": protocol,
        "frozen_transfer_protocol": transfer_protocol,
        "agent_policy_changed": False,
        "learning_inputs": learning_checks,
        "verification_inputs": checks,
        "allocation": {
            "catalogue": {"calls": 4, "cells": 8, "bytes": 256},
            "learning": {"calls": 2, "cells": 8, "bytes": 128},
            "verification": {"calls": 2, "cells": 8, "bytes": 128},
        },
        "development_environments_reused": True,
        "verification_inputs_redesigned_after_v2": True,
        "verification_targets": [
            "row_order",
            "collision_after_truncation",
            "unicode_normalization",
            "whitespace_and_empty_rows",
            "leading_zeros",
        ],
        "evaluation_seed": transfer_protocol["evaluation_seed"],
        "evaluation_tables": 200,
        "memory": "surviving_programs_and_observations",
        "evaluation_feedback_used_for_selection": False,
        "candidate_prior": "uniform_full_catalog_response_classes",
    })

    started = time.perf_counter()
    responses = []
    for index, program in enumerate(programs):
        responses.append([
            execute(program, test["input"]) for test in catalog
        ])
        if (index + 1) % 1000 == 0:
            print(f"Catalogue calcule : {index + 1}/{len(programs)}",
                  flush=True)
    signatures = [tuple(row) for row in responses]
    catalogue_seconds = time.perf_counter() - started

    arms = {
        "single": [
            i for i, p in enumerate(programs)
            if len(p["operations"]) <= 1
        ],
        "composed": list(range(len(programs))),
    }
    records = []
    saved_snapshots = {}

    for environment in protocol["environments"]:
        hidden = {
            "configuration_id": environment["configuration_id"],
            "operations": environment["operations"],
            "extra_operation_after_sequence":
                environment["extra_operation_after_sequence"],
        }
        for arm, initial in arms.items():
            started = time.perf_counter()
            survivors = list(initial)
            observations = []
            used = []
            calls = cells = text_bytes = 0

            def query(rows):
                nonlocal calls, cells, text_bytes
                calls += 1
                cells += sum(len(row) for row in rows)
                text_bytes += sum(
                    len(v.encode("utf-8")) for row in rows for v in row
                )
                if calls > 8 or cells > 24 or text_bytes > 512:
                    raise RuntimeError("Budget total depasse.")
                answer = execute(hidden, rows)
                observations.append({"input": rows, "output": answer})
                return answer

            while survivors and len(used) < 4:
                representatives = {}
                for i in survivors:
                    representatives.setdefault(signatures[i], i)
                if len(representatives) == 1:
                    break

                feasible = [
                    i for i, test in enumerate(catalog)
                    if i not in used
                    and cells + test["cells"] <= 8
                    and text_bytes + test["input_text_utf8_bytes"] <= 256
                ]
                if not feasible:
                    break

                def score(i):
                    groups = Counter(
                        responses[p][i] for p in representatives.values()
                    )
                    return (
                        sum(n*n for n in groups.values()),
                        catalog[i]["cells"],
                        catalog[i]["input_text_utf8_bytes"],
                        i,
                    )

                selected = min(feasible, key=score)
                if score(selected)[0] == len(representatives) ** 2:
                    break
                answer = query(catalog[selected]["input"])
                used.append(selected)
                survivors = [
                    i for i in survivors
                    if responses[i][selected] == answer
                ]

            # Deux observations d'apprentissage avant de figer.
            learning_observations = []
            for rows in learning_checks:
                if not survivors:
                    break
                answer = query(rows)
                survivors = [
                    i for i in survivors
                    if execute(programs[i], rows) == answer
                ]
                learning_observations.append({
                    "input": rows, "output": answer,
                    "remaining_programs": len(survivors),
                })

            # Figer les hypotheses avant les verifications.
            frozen = list(survivors)
            status = "outside_candidate_family" if not frozen else "provisional"
            verification = []
            for rows in checks if frozen else []:
                outputs = {execute(programs[i], rows) for i in frozen}
                answer = query(rows)
                unanimous = len(outputs) == 1
                matches = unanimous and next(iter(outputs)) == answer
                verification.append({
                    "input": rows,
                    "output": answer,
                    "unanimous": unanimous,
                    "matches": matches,
                })
                if answer not in outputs:
                    status = "revoked"
                elif not unanimous and status != "revoked":
                    status = "verification_ambiguous"

            memory = {
                "format": 1,
                "status": status,
                "programs": [programs[i] for i in frozen],
                "observations": observations,
                "verification": verification,
            }
            memory_path = output / (
                f"memory_{arm}_{environment['environment_id']:03d}.json"
            )
            persistence_inputs = [
                test["input"] for test in catalog
            ] + checks
            saved_snapshots[str(memory_path)] = (
                json.dumps(memory, sort_keys=True, ensure_ascii=False),
                [
                    execute(program, rows)
                    for program in memory["programs"]
                    for rows in persistence_inputs
                ],
            )
            write(memory_path, memory)

            records.append({
                "environment_id": environment["environment_id"],
                "group": environment["group"],
                "operations": environment["operations"],
                "configuration_id": environment["configuration_id"],
                "arm": arm,
                "status": status,
                "remaining_programs": len(frozen),
                "selected_test_ids": used,
                "learning_observations": learning_observations,
                "calls": calls,
                "cells": cells,
                "text_bytes": text_bytes,
                "search_and_verification_seconds":
                    time.perf_counter() - started,
                "memory_path": str(memory_path),
            })
            print(
                f"Env {environment['environment_id']:02d} | {arm} | "
                f"{status} | hypotheses={len(frozen)} | appels={calls}",
                flush=True,
            )

    # Tous les choix sont sauvegardes avant l'evaluation.
    write(output / "frozen_runs.json", records)

    cases = read(transfer_data / "evaluation_cases.json")
    write(output / "evaluation_cases.json", cases)

    environments = {
        e["environment_id"]: e for e in protocol["environments"]
    }
    candidate_evaluation_signatures = {
        tuple(execute(program, rows) for rows in cases)
        for program in programs
    }
    for record in records:
        memory_path = Path(record["memory_path"])
        restored = read(memory_path)
        environment = environments[record["environment_id"]]
        hidden = {
            "configuration_id": environment["configuration_id"],
            "operations": environment["operations"],
            "extra_operation_after_sequence":
                environment["extra_operation_after_sequence"],
        }
        true_signature = tuple(
            execute(hidden, rows) for rows in cases
        )
        record["matches_expanded_family_on_evaluation_tables"] = (
            true_signature in candidate_evaluation_signatures
        )
        answered = correct = 0

        # Les hypotheses rechargees seules servent aux predictions.
        if restored["status"] == "provisional":
            for rows in cases:
                outputs = {
                    execute(program, rows)
                    for program in restored["programs"]
                }
                if len(outputs) == 1:
                    answered += 1
                    prediction = next(iter(outputs))
                    correct += int(prediction == execute(hidden, rows))

        record.update(
            answered=answered,
            correct=correct,
            evaluation_cases=len(cases),
            memory_roundtrip_passed=(
                json.dumps(restored, sort_keys=True, ensure_ascii=False)
                == saved_snapshots[str(memory_path)][0]
                and [
                    execute(program, rows)
                    for program in restored["programs"]
                    for rows in (
                        [test["input"] for test in catalog] + checks
                    )
                ] == saved_snapshots[str(memory_path)][1]
            ),
        )

    if not all(r["memory_roundtrip_passed"] for r in records):
        raise RuntimeError("Echec du controle de persistance.")
    summaries = []
    for arm in arms:
        rows = [r for r in records if r["arm"] == arm]
        answered = sum(r["answered"] for r in rows)
        correct = sum(r["correct"] for r in rows)
        total = sum(r["evaluation_cases"] for r in rows)
        summary = {
            "arm": arm,
            "environments": len(rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "answered": answered,
            "correct": correct,
            "errors": answered - correct,
            "evaluation_cases": total,
            "coverage": answered / total,
            "total_calls": sum(r["calls"] for r in rows),
            "total_cells": sum(r["cells"] for r in rows),
            "total_text_bytes": sum(r["text_bytes"] for r in rows),
            "search_and_verification_seconds": sum(
                r["search_and_verification_seconds"] for r in rows
            ),
        }
        summaries.append(summary)
        print(json.dumps(summary))

    group_summaries = []
    for arm in arms:
        for group in sorted({r["group"] for r in records}):
            rows = [
                r for r in records
                if r["arm"] == arm and r["group"] == group
            ]
            answered = sum(r["answered"] for r in rows)
            correct = sum(r["correct"] for r in rows)
            total = sum(r["evaluation_cases"] for r in rows)
            report = {
                "arm": arm,
                "group": group,
                "environments": len(rows),
                "statuses": dict(Counter(r["status"] for r in rows)),
                "answered": answered,
                "correct": correct,
                "errors": answered - correct,
                "evaluation_cases": total,
                "coverage": answered / total,
                "accuracy_when_answered":
                    correct / answered if answered else None,
                "total_calls": sum(r["calls"] for r in rows),
                "total_cells": sum(r["cells"] for r in rows),
                "total_text_bytes": sum(r["text_bytes"] for r in rows),
                "no_expanded_candidate_matches_all_evaluation_tables":
                    sum(
                        not r[
                            "matches_expanded_family_on_evaluation_tables"
                        ]
                        for r in rows
                    ),
            }
            group_summaries.append(report)
            print(json.dumps(report))

    write(output / "results.json", {
        "group_summary": group_summaries,
        "protocol": read(output / "protocol.json"),
        "shared_catalogue_precomputation_seconds": catalogue_seconds,
        "summary": summaries,
        "results": records,
    })
    print(f"Resultats : {output / 'results.json'}")


if __name__ == "__main__":
    main()
