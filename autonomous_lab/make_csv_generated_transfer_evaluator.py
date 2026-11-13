import ast
import hashlib
import json
from pathlib import Path

root = Path("autonomous_lab")
prepared = root / "data/csv_generated_transfer_01"
frozen = json.loads(
    (prepared / "protocol.json").read_text(encoding="utf-8-sig")
)
fingerprints = dict(frozen["dependencies_sha256"])
fingerprints.update({
    str(prepared / name): digest
    for name, digest in frozen["prepared_files_sha256"].items()
})
for name, expected in fingerprints.items():
    path = Path(name)
    if not path.exists():
        raise SystemExit(f"Fichier manquant : {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Empreinte modifiee : {path}")

source = root / "evaluate_csv_counterexamples.py"
destination = root / "evaluate_csv_generated_transfer.py"
if destination.exists():
    raise SystemExit("Evaluateur existant : aucun remplacement.")

text = source.read_text(encoding="utf-8-sig")


def replace(old, new, count=1):
    global text
    actual = text.count(old)
    if actual != count:
        raise SystemExit(
            f"Structure inattendue : {old!r} trouve {actual} fois. "
            "Aucun evaluateur cree."
        )
    text = text.replace(old, new)


replace(
    '    root = Path("autonomous_lab")',
    '''    root = Path("autonomous_lab")
    comparison_data = root / "data/csv_generated_transfer_01"
    comparison_protocol = read(comparison_data / "protocol.json")
    fingerprints = dict(comparison_protocol["dependencies_sha256"])
    fingerprints.update({
        str(comparison_data / name): digest
        for name, digest in
        comparison_protocol["prepared_files_sha256"].items()
    })
    for name, expected in fingerprints.items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Empreinte modifiee : {name}")''',
)

replace(
    'protocol["environments"] = experiment["environments"]',
    '''protocol["environments"] = read(
        comparison_data / "environments.json"
    )''',
)
replace(
    'output = root / "runs/csv_counterexamples_01"',
    'output = root / "runs/csv_generated_transfer_01"',
)
replace(
    '"counterexample_protocol": experiment,',
    '''"counterexample_protocol": experiment,
        "frozen_comparison_protocol": comparison_protocol,''',
)
replace(
    '"evaluation_seed": 124001,',
    '"evaluation_seed": comparison_protocol["evaluation_seed"],',
)

replace(
    '''    banks = {
        "original_catalogue": (audit_catalog, original_responses),
        "generated_inputs": (generated, generated_responses),
    }''',
    '''    banks = {
        "generated_adaptive": (generated, generated_responses),
        "generated_fixed": (generated, generated_responses),
    }''',
)

replace(
    '''                selected = min(feasible, key=score)
                if score(selected)[0] == len(representatives) ** 2:
                    break''',
    '''                if arm == "generated_fixed":
                    available_fixed = [
                        i for i in comparison_protocol["fixed_sequence"]
                        if i in feasible
                    ]
                    if not available_fixed:
                        break
                    selected = available_fixed[0]
                else:
                    selected = min(feasible, key=score)
                    if score(selected)[0] == len(representatives) ** 2:
                        break''',
)

replace(
    '''    def execute(program, rows):
        return simulate(''',
    '''    def execute(program, rows):
        if program.get("extra_operation_after_sequence"):
            if (
                program["extra_operation_after_sequence"]
                != "reverse_columns"
                or program["operations"]
            ):
                raise RuntimeError("Operation externe inattendue.")
            rows = [list(reversed(row)) for row in rows]
        return simulate(''',
)

replace(
    '''        hidden = {
            "configuration_id": environment["configuration_id"],
            "operations": environment["operations"],
        }''',
    '''        hidden = {
            "configuration_id": environment["configuration_id"],
            "operations": environment["operations"],
            "extra_operation_after_sequence":
                environment["extra_operation_after_sequence"],
        }''',
    count=2,
)

replace(
    '"environment_id": environment["environment_id"],',
    '''"environment_id": environment["environment_id"],
                "group": environment["group"],''',
)

start_marker = "    rng = random.Random(124001)"
end_marker = '    write(output / "evaluation_cases.json", cases)'
if text.count(start_marker) != 1 or text.count(end_marker) != 1:
    raise SystemExit("Structure des tableaux inattendue.")
start = text.index(start_marker)
end = text.index(end_marker, start) + len(end_marker)
text = (
    text[:start]
    + '''    cases = read(comparison_data / "evaluation_cases.json")
    write(output / "evaluation_cases.json", cases)'''
    + text[end:]
)

replace(
    '    write(output / "results.json", {',
    '''    group_summary = []
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
                "total_calls": sum(r["calls"] for r in rows),
                "total_cells": sum(r["cells"] for r in rows),
                "total_text_bytes": sum(r["text_bytes"] for r in rows),
            }
            group_summary.append(report)
            print(json.dumps(report))

    paired = {}
    for row in records:
        paired.setdefault(row["environment_id"], {})[
            row["arm"]
        ] = {
            "answered": row["answered"],
            "correct": row["correct"],
            "errors": row["answered"] - row["correct"],
        }

    write(output / "results.json", {
        "group_summary": group_summary,
        "paired_environments": paired,''',
)

ast.parse(text)
destination.write_text(text, encoding="utf-8")
print("Empreintes verifiees.")
print(f"Cree : {destination}")
print("Aucune evaluation executee par le generateur.")
