import ast
import hashlib
import json
from pathlib import Path

root = Path("autonomous_lab")
data = root / "data/csv_counterexamples_01"
protocol = json.loads(
    (data / "protocol.json").read_text(encoding="utf-8-sig")
)
fingerprints = dict(protocol["dependencies_sha256"])
fingerprints[str(data / "probes.json")] = protocol["probes_sha256"]
for name, expected in fingerprints.items():
    path = Path(name)
    if not path.exists():
        raise SystemExit(f"Fichier manquant : {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Empreinte modifiee : {path}")

source = root / "acquire_csv_compositions_v3.py"
destination = root / "evaluate_csv_counterexamples.py"
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


if "import hashlib\n" not in text:
    text = "import hashlib\n" + text

replace(
    '    protocol = read(root / "data/csv_composition_01/protocol.json")',
    '''    prepared = root / "data/csv_counterexamples_01"
    experiment = read(prepared / "protocol.json")
    fingerprints = dict(experiment["dependencies_sha256"])
    fingerprints[str(prepared / "probes.json")] = (
        experiment["probes_sha256"]
    )
    for name, expected in fingerprints.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Empreinte modifiee : {name}")
    protocol = read(root / "data/csv_composition_01/protocol.json")
    protocol["environments"] = experiment["environments"]''',
)
replace(
    'output = root / "runs/csv_composition_03"',
    'output = root / "runs/csv_counterexamples_01"',
)
replace(
    '"source_protocol": protocol,',
    '''"source_protocol": protocol,
        "counterexample_protocol": experiment,''',
)

start_marker = "    arms = {"
end_marker = "    records = []"
if text.count(start_marker) != 1 or text.count(end_marker) != 1:
    raise SystemExit("Structure des groupes inattendue.")
start = text.index(start_marker)
end = text.index(end_marker, start)
text = (
    text[:start]
    + '''    audit_catalog = catalog
    original_responses = responses
    original_signatures = signatures
    generated = read(prepared / "probes.json")
    generated_started = time.perf_counter()
    generated_responses = []
    for index, program in enumerate(programs):
        generated_responses.append([
            execute(program, test["input"]) for test in generated
        ])
        if (index + 1) % 1000 == 0:
            print(
                f"Entrees generees calculees : "
                f"{index + 1}/{len(programs)}",
                flush=True,
            )
    generated_precomputation_seconds = (
        time.perf_counter() - generated_started
    )
    banks = {
        "original_catalogue": (audit_catalog, original_responses),
        "generated_inputs": (generated, generated_responses),
    }
    arms = {
        name: list(range(len(programs))) for name in banks
    }
'''
    + text[end:]
)

replace(
    '        for arm, initial in arms.items():',
    '''        for arm, initial in arms.items():
            catalog, responses = banks[arm]
            signatures = original_signatures''',
)

# Les controles de persistance utilisent le meme banc pour les deux bras.
replace(
    'test["input"] for test in catalog',
    'test["input"] for test in audit_catalog',
    count=2,
)

replace(
    'random.Random(121001)',
    'random.Random(124001)',
)
replace(
    '"evaluation_seed": 121001,',
    '"evaluation_seed": 124001,',
)

start_marker = "    forbidden = ("
end_marker = "    while len(cases) < 200:"
if text.count(start_marker) != 1 or text.count(end_marker) != 1:
    raise SystemExit("Structure des exclusions inattendue.")
start = text.index(start_marker)
end = text.index(end_marker, start)
text = (
    text[:start]
    + '''    forbidden = (
        [t["input"] for t in audit_catalog]
        + [t["input"] for t in generated]
        + checks + learning_checks
    )
    for previous_path in sorted(
        (root / "runs").glob("*/evaluation_cases.json")
    ):
        forbidden.extend(read(previous_path))
'''
    + text[end:]
)

replace(
    '"shared_catalogue_precomputation_seconds": catalogue_seconds,',
    '''"shared_catalogue_precomputation_seconds": catalogue_seconds,
        "generated_bank_precomputation_seconds":
            generated_precomputation_seconds,
        "same_expanded_candidate_family_in_both_arms": True,''',
)

ast.parse(text)
destination.write_text(text, encoding="utf-8")
print("Empreintes verifiees.")
print(f"Cree : {destination}")
print("Aucune evaluation executee par le generateur.")
