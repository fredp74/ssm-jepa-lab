import json
from collections import Counter
from pathlib import Path

from csv_environment import configurations, render, validate_rows
from csv_discovery import DiscoveryAgent, PROBES

CONFIGS = configurations()

def hidden_render(rows, config, variant):
    values = [list(row) for row in rows]
    options = dict(config)

    if variant == "pipe_separator":
        options["delimiter"] = "|"
    elif variant == "uppercase":
        values = [[value.upper() for value in row] for row in values]
    elif variant == "empty_as_NULL":
        values = [
            ["NULL" if value == "" else value for value in row]
            for row in values
        ]
    elif variant == "normalize_leading_zeros":
        values = [
            [
                str(int(value))
                if value.isascii() and value.isdigit()
                and len(value) > 1 and value.startswith("0")
                else value
                for value in row
            ]
            for row in values
        ]
    elif variant == "flatten_newlines":
        values = [
            [value.replace("\r\n", " ").replace("\n", " ") for value in row]
            for row in values
        ]

    result = render(values, options)
    if variant == "BOM_prefix":
        result = "\ufeff" + result
    return result

class HiddenTool:
    def __init__(self, config, variant):
        self._config = dict(config)
        self._variant = variant
        self.calls = []

    def query(self, rows):
        validate_rows(rows)
        if len(self.calls) >= 8:
            raise RuntimeError("Budget epuise.")
        answer = hidden_render(rows, self._config, self._variant)
        self.calls.append({
            "input": [list(row) for row in rows],
            "output": answer,
        })
        return answer

def score(agent, cases, expected):
    predictions = [agent.predict(case) for case in cases]
    answered = sum(value is not None for value in predictions)
    exact = sum(
        prediction is not None and prediction == answer
        for prediction, answer in zip(predictions, expected)
    )
    return {
        "remaining_candidates": len(agent.candidates),
        "answered": answered,
        "exact": exact,
        "cases": len(cases),
        "detected_unknown": len(agent.candidates) == 0,
        "all_predictions_correct": exact == len(cases),
    }

def main():
    output = Path("autonomous_lab/runs/csv_unknown_behavior_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    cases = json.loads(Path(
        "autonomous_lab/runs/csv_environment_01/reserved_cases.json"
    ).read_text(encoding="utf-8"))

    environments = [
        ("known", index, "known")
        for index in range(len(CONFIGS))
    ]
    variants = (
        "pipe_separator", "BOM_prefix", "uppercase",
        "empty_as_NULL", "normalize_leading_zeros", "flatten_newlines",
    )
    environments += [
        ("outside", index, variant)
        for index in (0, 7, 23, 47)
        for variant in variants
    ]

    results = []
    for group, config_id, variant in environments:
        config = CONFIGS[config_id]
        tool = HiddenTool(config, variant)
        agent = DiscoveryAgent()
        available = list(range(len(PROBES)))

        # L'evaluateur calcule ses references sans les communiquer a l'agent.
        expected = [
            hidden_render(case, config, variant) for case in cases
        ]
        if group == "outside":
            assert not any(
                all(
                    render(case, candidate) == answer
                    for case, answer in zip(cases, expected)
                )
                for candidate in CONFIGS
            ), "Ce comportement n'est pas distinct sur le jeu reserve."

        while len(agent.candidates) > 1 and available:
            probe = agent.choose_probe("active", available)
            available.remove(probe)
            agent.observe(PROBES[probe], tool.query(PROBES[probe]))

        discovery = score(agent, cases, expected)
        discovery["calls"] = len(tool.calls)

        # Verification supplementaire : toutes les sondes restantes.
        # Une contradiction suffit a rejeter la famille.
        while agent.candidates and available:
            probe = agent.choose_probe("active", available)
            available.remove(probe)
            agent.observe(PROBES[probe], tool.query(PROBES[probe]))

        audited = score(agent, cases, expected)
        audited["calls"] = len(tool.calls)
        results.append({
            "group": group,
            "base_configuration": config_id,
            "variant": variant,
            "discovery": discovery,
            "probe_audit": audited,
        })

    summaries = []
    for group in ("known", "outside"):
        for phase in ("discovery", "probe_audit"):
            rows = [row for row in results if row["group"] == group]
            summary = {
                "group": group,
                "phase": phase,
                "environments": len(rows),
                "detected_unknown": sum(
                    row[phase]["detected_unknown"] for row in rows
                ),
                "fully_correct_environments": sum(
                    row[phase]["all_predictions_correct"] for row in rows
                ),
                "incorrect_surviving_hypotheses": sum(
                    row[phase]["remaining_candidates"] > 0
                    and not row[phase]["all_predictions_correct"]
                    for row in rows
                ),
            }
            summaries.append(summary)
            print(summary)

    print("\nComportements hors famille apres audit :")
    for variant in variants:
        rows = [row for row in results if row["variant"] == variant]
        print(
            f"{variant} | inconnus detectes="
            f"{sum(row['probe_audit']['detected_unknown'] for row in rows)}/"
            f"{len(rows)}"
        )

    output.mkdir(parents=True)
    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "known_environments": 48,
                "outside_environments": 24,
                "outside_variants": list(variants),
                "maximum_calls": 8,
                "evaluation_cases_used_as_probes": False,
                "evaluator_feedback_given_to_agent": False,
                "variants_are_hand_designed": True,
            },
            "summaries": summaries,
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
