import itertools
import json
from pathlib import Path

from consolidation import Learner, RULES, INPUTS

def polynomial_table(coefficients):
    return tuple(
        sum(
            coefficient
            for mask, coefficient in enumerate(coefficients)
            if (index & mask) == mask
        ) % 2
        for index in range(8)
    )

def polynomial_family(max_degree):
    masks = [
        mask for mask in range(8)
        if bin(mask).count("1") <= max_degree
    ]
    tables = set()
    for bits in itertools.product((0, 1), repeat=len(masks)):
        coefficients = [0]*8
        for mask, bit in zip(masks, bits):
            coefficients[mask] = bit
        tables.add(polynomial_table(coefficients))
    return sorted(tables)

FAMILIES = [
    sorted(RULES),
    polynomial_family(2),
    polynomial_family(3),
]
assert len(FAMILIES[1]) == 128
assert len(FAMILIES[2]) == 256
assert set(FAMILIES[0]) <= set(FAMILIES[1])

def expression(table):
    # Transformation exacte de la table en polynome binaire.
    coefficients = list(table)
    for bit in (1, 2, 4):
        for index in range(8):
            if index & bit:
                coefficients[index] ^= coefficients[index ^ bit]

    terms = []
    for mask, coefficient in enumerate(coefficients):
        if not coefficient:
            continue
        variables = [
            name for bit, name in ((4, "x0"), (2, "x1"), (1, "x2"))
            if mask & bit
        ]
        terms.append("*".join(variables) if variables else "1")

    assert polynomial_table(coefficients) == tuple(table)
    return coefficients, " XOR ".join(terms) if terms else "0"

def acquire(ask):
    stage = 0
    observations = {}
    expansions = []
    selector = Learner()

    while True:
        candidates = [
            rule for rule in FAMILIES[stage]
            if all(rule[i] == y for i, y in observations.items())
        ]
        if not candidates:
            if stage == len(FAMILIES)-1:
                raise RuntimeError("Aucune regle compatible.")
            stage += 1
            expansions.append({
                "stage": stage,
                "after_questions": len(observations),
            })
            continue

        if len(observations) == 8:
            if len(candidates) != 1:
                raise RuntimeError("Audit complet sans regle unique.")
            return candidates[0], stage, observations, expansions

        if len(candidates) == 1:
            index = next(i for i in range(8) if i not in observations)
        else:
            index = selector.choose(candidates, set(observations))
        observations[index] = ask(index)

def main():
    output = Path("autonomous_lab/runs/grammar_expansion_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    details = []
    library = {}
    stage_counts = [0, 0, 0]

    for hidden in itertools.product((0, 1), repeat=8):
        def ask(index):
            return hidden[index]

        inferred, stage, observations, expansions = acquire(ask)
        assert inferred == hidden
        coefficients, readable = expression(inferred)
        signature = "".join(map(str, inferred))
        stage_counts[stage] += 1

        library[signature] = {
            "status": "verified_on_all_eight_inputs",
            "stage": stage,
            "coefficients": coefficients,
            "expression": readable,
        }
        details.append({
            "verified_table": signature,
            "stage": stage,
            "questions": len(observations),
            "expansions": expansions,
            "observations": [
                {"input": list(INPUTS[i]), "output": y}
                for i, y in observations.items()
            ],
        })

    library_path = output / "library.json"
    library_path.write_text(
        json.dumps(library, indent=2), encoding="utf-8"
    )
    restored = json.loads(library_path.read_text(encoding="utf-8"))
    for signature, rule in restored.items():
        assert "".join(map(str, polynomial_table(
            rule["coefficients"]
        ))) == signature

    summary = {
        "functions_verified": len(details),
        "final_stage_counts": stage_counts,
        "total_questions": sum(row["questions"] for row in details),
        "rules_restored_and_checked": len(restored),
    }
    (output / "results.json").write_text(
        json.dumps({
            "config": {
                "family_sizes": [len(family) for family in FAMILIES],
                "expansion_trigger": "no_compatible_rule",
                "observations_preserved_on_expansion": True,
                "acceptance": "full_domain_audit",
                "noise": False,
                "extensions_are_programmed": True,
            },
            "summary": summary,
            "details": details,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"Fonctions verifiees : {len(details)}/256")
    print(f"Grammaire initiale : {stage_counts[0]}")
    print(f"Extension degre 2 : {stage_counts[1]}")
    print(f"Extension degre 3 : {stage_counts[2]}")
    print(f"Questions totales : {summary['total_questions']}")
    print(f"Regles rechargees et verifiees : {len(restored)}")
    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
