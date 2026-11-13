import ast
import hashlib
import json
from pathlib import Path

root = Path("autonomous_lab")
data = root / "data/csv_transfer_01"
protocol = json.loads(
    (data / "protocol.json").read_text(encoding="utf-8-sig")
)

fingerprints = dict(protocol["agent_and_dependencies_sha256"])
fingerprints.update({
    str(data / name): digest
    for name, digest in protocol["prepared_files_sha256"].items()
})
for name, expected in fingerprints.items():
    path = Path(name)
    if not path.exists():
        raise SystemExit(f"Fichier manquant : {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Empreinte modifiee : {path}")

source = root / "acquire_csv_compositions_v3.py"
destination = root / "evaluate_csv_transfer.py"
if destination.exists():
    raise SystemExit("Evaluateur existant : aucun remplacement.")

text = source.read_text(encoding="utf-8-sig")


def replace_exact(old, new, expected=1):
    global text
    count = text.count(old)
    if count != expected:
        raise SystemExit(
            f"Structure inattendue : {old!r} trouve {count} fois. "
            "Aucun evaluateur cree."
        )
    text = text.replace(old, new)


replace_exact(
    '    protocol = read(root / "data/csv_composition_01/protocol.json")',
    '''    transfer_data = root / "data/csv_transfer_01"
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
    )''',
)

replace_exact(
    'output = root / "runs/csv_composition_03"',
    'output = root / "runs/csv_transfer_01"',
)

# Import requis pour verifier les empreintes pendant l'execution.
if "import hashlib\n" not in text:
    text = "import hashlib\n" + text

replace_exact(
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

hidden_old = '''        hidden = {
            "configuration_id": environment["configuration_id"],
            "operations": environment["operations"],
        }'''
hidden_new = '''        hidden = {
            "configuration_id": environment["configuration_id"],
            "operations": environment["operations"],
            "extra_operation_after_sequence":
                environment["extra_operation_after_sequence"],
        }'''
replace_exact(hidden_old, hidden_new, expected=2)

replace_exact(
    '"source_protocol": protocol,',
    '''"source_protocol": protocol,
        "frozen_transfer_protocol": transfer_protocol,
        "agent_policy_changed": False,''',
)

replace_exact(
    '"evaluation_seed": 121001,',
    '"evaluation_seed": transfer_protocol["evaluation_seed"],',
)

replace_exact(
    '"environment_id": environment["environment_id"],',
    '''"environment_id": environment["environment_id"],
                "group": environment["group"],''',
)

# Remplacer la generation de tableaux par les fichiers deja figes.
start_marker = "    rng = random.Random(121001)"
end_marker = '    write(output / "evaluation_cases.json", cases)'
if text.count(start_marker) != 1 or text.count(end_marker) != 1:
    raise SystemExit("Bloc des tableaux inattendu : aucun evaluateur cree.")
start = text.index(start_marker)
end = text.index(end_marker, start) + len(end_marker)
text = (
    text[:start]
    + '''    cases = read(transfer_data / "evaluation_cases.json")
    write(output / "evaluation_cases.json", cases)'''
    + text[end:]
)

# Ce diagnostic intervient apres la sauvegarde des choix de l'agent.
replace_exact(
    '    for record in records:',
    '''    candidate_evaluation_signatures = {
        tuple(execute(program, rows) for rows in cases)
        for program in programs
    }
    for record in records:''',
)

replace_exact(
    '        answered = correct = 0',
    '''        true_signature = tuple(
            execute(hidden, rows) for rows in cases
        )
        record["matches_expanded_family_on_evaluation_tables"] = (
            true_signature in candidate_evaluation_signatures
        )
        answered = correct = 0''',
)

replace_exact(
    '    write(output / "results.json", {',
    '''    group_summaries = []
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
        "group_summary": group_summaries,''',
)

ast.parse(text)
destination.write_text(text, encoding="utf-8")
print("Empreintes verifiees.")
print(f"Cree : {destination}")
print("Aucune evaluation executee par ce generateur.")
