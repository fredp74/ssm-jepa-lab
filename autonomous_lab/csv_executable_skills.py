import hashlib
import json
from pathlib import Path

from csv_environment import configurations, validate_rows
from csv_policy_behaviors import simulate

def execute(program, rows):
    validate_rows(rows)
    return simulate(
        rows,
        program["csv_config"],
        program["operations"],
    )

def save_skill(path, program, observations, checks):
    # Les controles ne constituent pas une preuve generale.
    data = {
        "format": 1,
        "status": "provisional",
        "representation": "programmed_transformation_then_csv",
        "program": program,
        "observations": observations,
        "independent_checks": checks,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)

def load_skill(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data["format"] != 1
        or data["representation"] != "programmed_transformation_then_csv"
    ):
        raise ValueError("Competence incompatible.")
    return data

def main():
    source = Path("autonomous_lab/data/csv_ssm_policy_01")
    output = Path("autonomous_lab/data/csv_skill_acquisition_01")
    if output.exists():
        raise SystemExit("Preparation deja presente : aucun remplacement.")

    protocol_raw = (source / "protocol.json").read_bytes()
    protocol = json.loads(protocol_raw.decode("utf-8"))
    catalog_raw = (source / "catalog.json").read_bytes()
    catalog = json.loads(catalog_raw.decode("utf-8"))

    configs = configurations()
    operations = ["identity"] + protocol[
        "training_behaviors"
    ]["individual_operations"]

    programs = []
    classes = {}
    for operation in operations:
        for config_id, config in enumerate(configs):
            program = {
                "csv_config": config,
                "operations": [] if operation == "identity" else [operation],
            }
            responses = [
                execute(program, test["input"]) for test in catalog
            ]
            signature = hashlib.sha256(
                json.dumps(responses, ensure_ascii=False).encode("utf-8")
            ).hexdigest()

            program_id = len(programs)
            programs.append({
                "program_id": program_id,
                "configuration_id": config_id,
                "program": program,
                "catalog_response_signature": signature,
            })
            classes.setdefault(signature, []).append(program_id)

    output.mkdir(parents=True)
    (output / "programs.json").write_text(
        json.dumps(programs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "observable_classes.json").write_text(
        json.dumps(classes, indent=2), encoding="utf-8",
    )

    # Controle de sauvegarde uniquement : programme choisi explicitement,
    # sans pretendre qu'un agent l'a deja decouvert.
    example = next(
        item["program"] for item in programs
        if item["program"]["operations"] == ["normalize_leading_zeros"]
    )
    probe = [["00053", "plain_example"]]
    expected = execute(example, probe)
    check_path = output / "persistence_check.json"
    save_skill(
        check_path,
        example,
        [{"input": probe, "output": expected}],
        [],
    )
    restored = load_skill(check_path)
    assert execute(restored["program"], probe) == expected

    summary = {
        "candidate_programs": len(programs),
        "distinct_catalog_response_classes": len(classes),
        "persistence_check_passed": True,
        "agent_acquisition_tested": False,
        "neural_training": False,
        "reserved_compositions_used": False,
        "source_protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "catalog_sha256": hashlib.sha256(catalog_raw).hexdigest(),
    }
    (output / "protocol.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Preparation : {output}")

if __name__ == "__main__":
    main()
