import json
from collections import defaultdict
from pathlib import Path

from csv_environment import (
    BudgetedTool, configurations, render, evaluate_predictions
)

CONFIGS = configurations()
PROBES = [
    [["probe_A"]],
    [["probe_B", "probe_C"], ["probe_D", "probe_E"]],
    [["  padded_F  ", "\tspaced_G\t"]],
    [['quoted_"H"', 'another_"I"']],
    [["slash\\J", 'slash\\"K']],
    [["line_L\nline_M", "line_N\r\nline_O"]],
    [["comma,P", "semi;Q", "tab\tR"]],
    [
        ["  mix,S;T\tU  ", 'quote"V', "slash\\W", ""],
        ["line_X\nline_Y", "line_Z\r\nline_AA", "été_BB", " CC "],
    ],
]

class DiscoveryAgent:
    def __init__(self, observations=None):
        self.observations = list(observations or [])
        self.candidates = [
            index for index, config in enumerate(CONFIGS)
            if all(
                render(item["input"], config) == item["output"]
                for item in self.observations
            )
        ]

    def choose_probe(self, policy, available):
        if policy == "fixed":
            return available[0]

        def score(probe_id):
            groups = defaultdict(int)
            for candidate in self.candidates:
                result = render(PROBES[probe_id], CONFIGS[candidate])
                groups[result] += 1
            # Minimiser le nombre espere de candidats restants,
            # sous une distribution uniforme sur les hypotheses.
            return sum(size*size for size in groups.values())

        return min(available, key=score)

    def observe(self, rows, output):
        observation = {
            "input": [list(row) for row in rows],
            "output": output,
        }
        self.observations.append(observation)
        self.candidates = [
            candidate for candidate in self.candidates
            if render(rows, CONFIGS[candidate]) == output
        ]

    def predict(self, rows):
        if not self.candidates:
            return None
        possible = {
            render(rows, CONFIGS[candidate])
            for candidate in self.candidates
        }
        return next(iter(possible)) if len(possible) == 1 else None

    def save(self, path):
        path.write_text(
            json.dumps({
                "format": 1,
                "observations": self.observations,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["format"] != 1:
            raise ValueError("Memoire incompatible.")
        return cls(data["observations"])

def main():
    output = Path("autonomous_lab/runs/csv_discovery_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    # Ce fichier appartient uniquement a l'evaluateur.
    cases = json.loads(Path(
        "autonomous_lab/runs/csv_environment_01/reserved_cases.json"
    ).read_text(encoding="utf-8"))

    # Verifier qu'aucune sonde n'est un tableau de test complet.
    assert not any(probe == case for probe in PROBES for case in cases)

    output.mkdir(parents=True)
    results = []

    for policy in ("fixed", "active"):
        for config_id, hidden_config in enumerate(CONFIGS):
            tool = BudgetedTool(hidden_config, budget=8)
            agent = DiscoveryAgent()
            available = list(range(len(PROBES)))
            trace = []

            while len(agent.candidates) > 1 and available:
                probe_id = agent.choose_probe(policy, available)
                available.remove(probe_id)
                response = tool.query(PROBES[probe_id])
                agent.observe(PROBES[probe_id], response)
                trace.append({
                    "probe_id": probe_id,
                    "remaining_candidates": len(agent.candidates),
                })

            path = output / f"{policy}_{config_id:02d}_memory.json"
            agent.save(path)
            restored = DiscoveryAgent.load(path)
            assert restored.candidates == agent.candidates

            # Predictions sans appels supplementaires a l'outil.
            calls_before = len(tool.calls)
            predictions = [restored.predict(case) for case in cases]
            score = evaluate_predictions(cases, predictions, hidden_config)
            assert len(tool.calls) == calls_before

            results.append({
                "policy": policy,
                "environment_id": config_id,
                "calls": len(tool.calls),
                "remaining_candidates": len(restored.candidates),
                "trace": trace,
                **score,
            })

        rows = [row for row in results if row["policy"] == policy]
        print(
            f"{policy} | appels moyens="
            f"{sum(row['calls'] for row in rows)/len(rows):.2f} | "
            f"configurations identifiees="
            f"{sum(row['remaining_candidates']==1 for row in rows)}/48 | "
            f"predictions exactes={sum(row['exact'] for row in rows)}/4800 | "
            f"abstentions={sum(row['abstentions'] for row in rows)}",
            flush=True,
        )

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "environments": len(CONFIGS),
                "maximum_calls": 8,
                "probe_bank": PROBES,
                "active_objective": "minimize_expected_candidate_count",
                "known_family": True,
                "uniform_candidate_prior": True,
                "neural_training": False,
                "prediction_after_memory_reload": True,
                "evaluation_tool_calls": 0,
            },
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
