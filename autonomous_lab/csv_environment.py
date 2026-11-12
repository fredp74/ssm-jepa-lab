import csv
import hashlib
import io
import itertools
import json
import random
from pathlib import Path

def configurations():
    return [
        {
            "delimiter": delimiter,
            "quote_all": quote_all,
            "doublequote": doublequote,
            "line_ending": line_ending,
            "trim_fields": trim_fields,
        }
        for delimiter, quote_all, doublequote, line_ending, trim_fields
        in itertools.product(
            (",", ";", "\t"),
            (False, True),
            (False, True),
            ("\n", "\r\n"),
            (False, True),
        )
    ]

def validate_rows(rows):
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, list) and row for row in rows)
    ):
        raise ValueError("Un tableau non vide est requis.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Les lignes doivent avoir la meme largeur.")
    if any(not isinstance(value, str) for row in rows for value in row):
        raise ValueError("Chaque cellule doit etre une chaine.")

def render(rows, config):
    validate_rows(rows)
    values = [
        [value.strip() if config["trim_fields"] else value for value in row]
        for row in rows
    ]
    output = io.StringIO(newline="")
    writer = csv.writer(
        output,
        delimiter=config["delimiter"],
        quotechar='"',
        quoting=csv.QUOTE_ALL if config["quote_all"] else csv.QUOTE_MINIMAL,
        doublequote=config["doublequote"],
        escapechar=None if config["doublequote"] else "\\",
        lineterminator=config["line_ending"],
    )
    writer.writerows(values)
    return output.getvalue()

class BudgetedTool:
    def __init__(self, config, budget=8):
        self._config = dict(config)
        self._budget = budget
        self.calls = []

    def query(self, rows):
        validate_rows(rows)
        if len(self.calls) >= self._budget:
            raise RuntimeError("Budget d'exploration epuise.")
        result = render(rows, self._config)
        self.calls.append({
            "input": [list(row) for row in rows],
            "output": result,
        })
        return result

def make_evaluation_cases():
    # Cas d'evaluation : ne pas les transmettre a l'agent comme sondes.
    cases = [
        [[""]],
        [["", "", ""]],
        [["a,b", "c;d", "e\tf"]],
        [['Il dit "oui"', "normal"]],
        [["ligne 1\nligne 2", "fin"]],
        [["ligne 1\r\nligne 2", "fin"]],
        [["  espace  ", "\ttexte\t"]],
        [["été", "東京", "🙂"]],
        [["a\\b", 'c\\"d']],
        [["a", "b"], ["c", ""]],
    ]
    rng = random.Random(101001)
    values = [
        "", "alpha", "42", "001", " a ", "   ", "a,b", "a;b", "a\tb",
        '"', 'a"b', "a\\b", "x\ny", "x\r\ny", "été", "東京", "🙂",
    ]
    while len(cases) < 100:
        height = rng.randint(1, 4)
        width = rng.randint(1, 4)
        case = [
            [rng.choice(values) for _ in range(width)]
            for _ in range(height)
        ]
        if case not in cases:
            cases.append(case)
    return cases

def evaluate_predictions(cases, predictions, config):
    if len(cases) != len(predictions):
        raise ValueError("Nombre de predictions incorrect.")
    answered = correct = 0
    for rows, prediction in zip(cases, predictions):
        if prediction is None:
            continue
        if not isinstance(prediction, str):
            raise ValueError("Une prediction doit etre une chaine ou None.")
        answered += 1
        correct += prediction == render(rows, config)
    return {
        "cases": len(cases),
        "answered": answered,
        "exact": correct,
        "abstentions": len(cases)-answered,
    }

def main():
    output = Path("autonomous_lab/runs/csv_environment_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    configs = configurations()
    cases = make_evaluation_cases()
    checks = 0

    for config in configs:
        for rows in cases:
            encoded = render(rows, config)
            decoded = list(csv.reader(
                io.StringIO(encoded, newline=""),
                delimiter=config["delimiter"],
                quotechar='"',
                doublequote=config["doublequote"],
                escapechar=None if config["doublequote"] else "\\",
                strict=True,
            ))
            expected = [
                [value.strip() if config["trim_fields"] else value for value in row]
                for row in rows
            ]
            if decoded != expected:
                raise AssertionError({
                    "config": config, "input": rows,
                    "decoded": decoded, "expected": expected,
                })
            checks += 1

    # Verifier le comptage des appels et la protection du journal.
    tool = BudgetedTool(configs[0], budget=2)
    probe = [["probe"]]
    tool.query(probe)
    probe[0][0] = "modified"
    assert tool.calls[0]["input"] == [["probe"]]
    tool.query([["second"]])
    try:
        tool.query([["third"]])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Le budget n'est pas applique.")

    output.mkdir(parents=True)
    serialized_cases = json.dumps(
        cases, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    summary = {
        "configurations": len(configs),
        "reserved_cases": len(cases),
        "roundtrip_checks_passed": checks,
        "budget_check_passed": True,
        "input_log_copy_check_passed": True,
        "evaluation_cases_sha256": hashlib.sha256(serialized_cases).hexdigest(),
        "agent_evaluated": False,
    }
    (output / "reserved_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
