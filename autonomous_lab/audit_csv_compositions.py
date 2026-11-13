import hashlib
import json
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
    source = root / "runs/csv_composition_01"
    output = root / "runs/csv_composition_audit_01"
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    original = read(source / "results.json")
    cases = read(source / "evaluation_cases.json")
    catalog = read(root / "data/csv_ssm_policy_01/catalog.json")
    configs = configurations()

    def execute(program, rows):
        return simulate(
            rows,
            configs[program["configuration_id"]],
            program["operations"],
        )

    def prediction_digest(programs, inputs):
        digest = hashlib.sha256()
        for program in programs:
            for rows in inputs:
                encoded = json.dumps(
                    execute(program, rows), ensure_ascii=False
                ).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        return digest.hexdigest()

    def measure(programs, hidden):
        if not programs:
            return {"answered": 0, "correct": 0}
        answered = correct = 0
        for rows in cases:
            predictions = {execute(p, rows) for p in programs}
            if len(predictions) == 1:
                answered += 1
                correct += int(
                    next(iter(predictions)) == execute(hidden, rows)
                )
        return {"answered": answered, "correct": correct}

    output.mkdir(parents=True)
    write(output / "protocol.json", {
        "retrospective_diagnostic": True,
        "original_results_modified": False,
        "new_tool_calls": 0,
        "evaluation_tables_reused": True,
        "verification_feedback_reused_for_candidate_filtering": True,
        "persistence_scope": (
            "new serialization of existing memories; "
            "not proof of the historical original roundtrip"
        ),
        "source_results_sha256": hashlib.sha256(
            (source / "results.json").read_bytes()
        ).hexdigest(),
    })

    records = []
    for old in original["results"]:
        filename = (
            f"memory_{old['arm']}_{old['environment_id']:03d}.json"
        )
        memory = read(source / filename)
        inputs = (
            [t["input"] for t in catalog]
            + [check["input"] for check in memory["verification"]]
            + cases
        )

        # Predire AVANT la nouvelle sauvegarde.
        before_digest = prediction_digest(memory["programs"], inputs)
        copied_path = output / filename
        write(copied_path, memory)

        # Recharger et predire APRES la sauvegarde.
        restored = read(copied_path)
        after_digest = prediction_digest(restored["programs"], inputs)
        persistence_passed = (
            restored == memory and before_digest == after_digest
        )
        if not persistence_passed:
            raise RuntimeError(f"Persistance incorrecte : {filename}")

        record = {
            "arm": old["arm"],
            "environment_id": old["environment_id"],
            "operations": old["operations"],
            "original_status": old["status"],
            "payload_roundtrip_passed": restored == memory,
            "prediction_roundtrip_passed": before_digest == after_digest,
            "prediction_digest_before": before_digest,
            "prediction_digest_after": after_digest,
            "test_inputs_per_program": len(inputs),
        }

        if (
            old["arm"] == "composed"
            and old["status"] == "verification_ambiguous"
        ):
            candidates = list(restored["programs"])
            trace = []
            for check in restored["verification"]:
                predicted_outputs = {
                    execute(p, check["input"]) for p in candidates
                }
                before_count = len(candidates)
                candidates = [
                    p for p in candidates
                    if execute(p, check["input"]) == check["output"]
                ]
                trace.append({
                    "candidates_before": before_count,
                    "distinct_predictions": len(predicted_outputs),
                    "candidates_after": len(candidates),
                })

            hidden = {
                "configuration_id": old["configuration_id"],
                "operations": old["operations"],
            }
            record.update({
                "candidates_before": len(restored["programs"]),
                "candidates_after": len(candidates),
                "verification_filter_trace": trace,
                "consensus_before_filtering": measure(
                    restored["programs"], hidden
                ),
                "consensus_after_filtering": measure(candidates, hidden),
                "filtered_programs": candidates,
            })

        records.append(record)
        print(
            f"Env {old['environment_id']:02d} | {old['arm']} | "
            f"persistance=OK",
            flush=True,
        )

    ambiguous = [r for r in records if "candidates_after" in r]
    before = sum(
        r["consensus_before_filtering"]["answered"] for r in ambiguous
    )
    after = sum(
        r["consensus_after_filtering"]["answered"] for r in ambiguous
    )
    correct = sum(
        r["consensus_after_filtering"]["correct"] for r in ambiguous
    )
    summary = {
        "memories_checked": len(records),
        "prediction_roundtrips_passed": sum(
            r["prediction_roundtrip_passed"] for r in records
        ),
        "ambiguous_environments_examined": len(ambiguous),
        "remaining_candidate_counts": dict(Counter(
            r["candidates_after"] for r in ambiguous
        )),
        "evaluation_questions_in_ambiguous_subset":
            len(ambiguous) * len(cases),
        "diagnostic_consensus_answers_before_filtering": before,
        "diagnostic_consensus_answers_after_filtering": after,
        "diagnostic_correct_after_filtering": correct,
        "diagnostic_errors_after_filtering": after - correct,
        "new_tool_calls": 0,
    }
    write(output / "results.json", {
        "protocol": read(output / "protocol.json"),
        "summary": summary,
        "results": records,
    })
    print(json.dumps(summary, indent=2))
    print(f"Resultats : {output / 'results.json'}")


if __name__ == "__main__":
    main()
