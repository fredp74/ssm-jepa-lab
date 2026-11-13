import hashlib
import json
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
    prepared = root / "data/csv_independent_checks_01"
    protocol = read(prepared / "protocol.json")
    fingerprints = dict(protocol["dependencies_sha256"])
    fingerprints.update({
        str(prepared / name): digest
        for name, digest in protocol["prepared_files_sha256"].items()
    })
    for name, expected in fingerprints.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Empreinte modifiee : {name}")

    output = root / "runs/csv_independent_checks_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    environments = read(prepared / "environments.json")
    banks = read(prepared / "check_banks.json")
    probes = read(root / "data/csv_counterexamples_01/probes.json")
    catalog = read(root / "data/csv_ssm_policy_01/catalog.json")
    grammar = read(root / "data/csv_composition_01/protocol.json")
    configs = configurations()
    programs = [
        {"configuration_id": c, "operations": sequence}
        for c in range(len(configs))
        for sequence in grammar["candidate_operation_sequences"]
    ]

    def execute(program, rows):
        extra = program.get("extra_operation_after_sequence")
        if extra:
            if extra != "rotate_columns_left" or program["operations"]:
                raise RuntimeError("Operation externe inattendue.")
            rows = [row[1:] + row[:1] for row in rows]
        return simulate(
            rows, configs[program["configuration_id"]],
            program["operations"],
        )

    learning = [
        [["A", "00042"], ["a", "00042"]],
        [["  ", ""], ["e\u0301_verify ", "7.25"]],
    ]
    fixed_checks = [
        [
            ["z_verify"], ["a_verify"],
            ["abcdefghijklmnop_X"], ["abcdefghijklmnop_Y"],
        ],
        [["e\u0301_check "], ["é_check "], ["   "], ["00048"]],
    ]
    checks_by_arm = {"fixed": fixed_checks}
    checks_by_arm.update({
        f"independent_{bank['seed']}": bank["inputs"] for bank in banks
    })

    output.mkdir(parents=True)
    write(output / "protocol.json", protocol)
    started = time.perf_counter()
    responses = []
    for i, program in enumerate(programs):
        responses.append([
            execute(program, test["input"]) for test in probes
        ])
        if (i + 1) % 1000 == 0:
            print(f"Precalcul : {i + 1}/{len(programs)}", flush=True)
    precomputation_seconds = time.perf_counter() - started

    # Classes identiques a celles du catalogue original.
    signatures = [
        tuple(execute(program, test["input"]) for test in catalog)
        for program in programs
    ]
    records = []
    shared_learning_seconds = 0.0

    def digest_predictions(ids, inputs):
        digest = hashlib.sha256()
        for i in ids:
            for rows in inputs:
                value = json.dumps(
                    execute(programs[i], rows), ensure_ascii=False
                ).encode("utf-8")
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
        return digest.hexdigest()

    for environment in environments:
        started = time.perf_counter()
        survivors = list(range(len(programs)))
        observations = []
        actions = []
        calls = cells = size = 0

        def observe(rows):
            nonlocal calls, cells, size
            calls += 1
            cells += sum(len(row) for row in rows)
            size += sum(len(v.encode("utf-8")) for row in rows for v in row)
            answer = execute(environment, rows)
            observations.append({"input": rows, "output": answer})
            return answer

        for _ in range(4):
            if not survivors or len({signatures[i] for i in survivors}) == 1:
                break
            feasible = [
                i for i in protocol["fixed_discovery_sequence"]
                if i not in actions
                and cells + probes[i]["cells"] <= 8
                and size + probes[i]["input_text_utf8_bytes"] <= 256
            ]
            if not feasible:
                break
            selected = feasible[0]
            answer = observe(probes[selected]["input"])
            actions.append(selected)
            survivors = [
                i for i in survivors if responses[i][selected] == answer
            ]

        for rows in learning:
            if not survivors:
                break
            answer = observe(rows)
            survivors = [
                i for i in survivors
                if execute(programs[i], rows) == answer
            ]

        shared_learning_seconds += time.perf_counter() - started
        frozen = tuple(survivors)

        for arm, checks in checks_by_arm.items():
            status = "provisional" if frozen else "outside_candidate_family"
            verification = []
            arm_calls, arm_cells, arm_bytes = calls, cells, size
            started = time.perf_counter()
            for rows in checks if frozen else []:
                predictions = {execute(programs[i], rows) for i in frozen}
                answer = execute(environment, rows)
                arm_calls += 1
                arm_cells += sum(len(row) for row in rows)
                arm_bytes += sum(
                    len(v.encode("utf-8")) for row in rows for v in row
                )
                verification.append({
                    "input": rows, "output": answer,
                    "distinct_predictions": len(predictions),
                    "unanimous_match":
                        len(predictions) == 1 and answer in predictions,
                })
                if answer not in predictions:
                    status = "revoked"
                elif len(predictions) != 1 and status != "revoked":
                    status = "verification_ambiguous"
            verification_seconds = time.perf_counter() - started
            if arm_calls > 8 or arm_cells > 24 or arm_bytes > 512:
                raise RuntimeError("Budget depasse.")

            memory = {
                "status": status,
                "candidate_ids": list(frozen),
                "programs": [programs[i] for i in frozen],
                "learning_observations": observations,
                "verification": verification,
            }
            inputs = [t["input"] for t in catalog] + checks
            before_digest = digest_predictions(frozen, inputs)
            memory_path = output / (
                f"memory_{environment['environment_id']:03d}_{arm}.json"
            )
            write(memory_path, memory)
            restored = read(memory_path)
            after_hash = hashlib.sha256()
            for program in restored["programs"]:
                for rows in inputs:
                    value = json.dumps(
                        execute(program, rows), ensure_ascii=False
                    ).encode("utf-8")
                    after_hash.update(len(value).to_bytes(8, "big"))
                    after_hash.update(value)
            passed = restored == memory and before_digest == after_hash.hexdigest()
            if not passed:
                raise RuntimeError("Echec de persistance.")
            records.append({
                "environment_id": environment["environment_id"],
                "group": environment["group"],
                "operations": environment["operations"],
                "arm": arm, "status": status,
                "candidate_ids": list(frozen),
                "selected_test_ids": actions,
                "calls": arm_calls, "cells": arm_cells,
                "text_bytes": arm_bytes,
                "verification_seconds": verification_seconds,
                "prediction_roundtrip_passed": passed,
            })
        print(
            f"Environnement termine : {environment['environment_id'] + 1}"
            f"/{len(environments)}",
            flush=True,
        )

    # Toutes les decisions sont figees avant cette evaluation.
    write(output / "frozen_runs.json", records)
    cases = read(prepared / "evaluation_cases.json")
    predictions = [
        tuple(execute(program, rows) for rows in cases)
        for program in programs
    ]
    family_signatures = set(predictions)
    expected = {
        e["environment_id"]: tuple(execute(e, rows) for rows in cases)
        for e in environments
    }
    for record in records:
        truth = expected[record["environment_id"]]
        answered = correct = 0
        if record["status"] == "provisional":
            for k, answer in enumerate(truth):
                outputs = {
                    predictions[i][k] for i in record["candidate_ids"]
                }
                if len(outputs) == 1:
                    answered += 1
                    correct += int(next(iter(outputs)) == answer)
        record.update(
            answered=answered, correct=correct,
            errors=answered - correct,
            evaluation_cases=len(cases),
            matches_candidate_family_on_evaluation_tables=(
                truth in family_signatures
            ),
        )

    fixed = {
        r["environment_id"]: r for r in records if r["arm"] == "fixed"
    }
    summaries = []
    for arm in checks_by_arm:
        for group in ["ALL"] + sorted({e["group"] for e in environments}):
            rows = [
                r for r in records
                if r["arm"] == arm and (group == "ALL" or r["group"] == group)
            ]
            total = sum(r["evaluation_cases"] for r in rows)
            report = {
                "arm": arm, "group": group, "environments": len(rows),
                "statuses": dict(Counter(r["status"] for r in rows)),
                "answered": sum(r["answered"] for r in rows),
                "correct": sum(r["correct"] for r in rows),
                "errors": sum(r["errors"] for r in rows),
                "accepted_environments_with_errors":
                    sum(r["errors"] > 0 for r in rows),
                "evaluation_cases": total,
                "coverage": sum(r["answered"] for r in rows) / total,
                "fixed_acceptances_with_errors_rejected":
                    sum(
                        fixed[r["environment_id"]]["errors"] > 0
                        and r["status"] != "provisional"
                        for r in rows
                    ),
                "fixed_correct_answers_lost_by_environment":
                    sum(
                        max(0, fixed[r["environment_id"]]["correct"] - r["correct"])
                        for r in rows
                    ),
                "total_calls": sum(r["calls"] for r in rows),
                "total_cells": sum(r["cells"] for r in rows),
                "total_text_bytes": sum(r["text_bytes"] for r in rows),
            }
            summaries.append(report)
            if group == "ALL":
                print(json.dumps(report))

    write(output / "results.json", {
        "protocol": protocol,
        "summary": summaries,
        "shared_precomputation_seconds": precomputation_seconds,
        "shared_learning_seconds": shared_learning_seconds,
        "results": records,
    })
    print(f"Resultats : {output / 'results.json'}")


if __name__ == "__main__":
    main()
