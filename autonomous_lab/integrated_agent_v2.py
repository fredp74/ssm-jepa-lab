import json
from pathlib import Path

from budgeted_five_bits import FAMILY
from local_repair import repair_family

BASE = list(FAMILY)
REPAIR = repair_family()

def compatible(rule, observations):
    return all(
        rule[int(index)] == value
        for index, value in observations.items()
    )

class Agent:
    def __init__(self, state=None):
        self.state = state or {
            "format": 1,
            "epoch": 0,
            "audit_cursor": 0,
            "stage": "base",
            "observations": {},
            "archives": [],
            "events": [],
        }

    def candidates(self):
        observations = self.state["observations"]
        family = BASE if self.state["stage"] == "base" else REPAIR
        return [
            rule for rule in family
            if compatible(rule, observations)
        ]

    def prediction(self, index):
        observations = self.state["observations"]
        if str(index) in observations:
            return observations[str(index)], "memory"

        candidates = self.candidates()
        if not candidates:
            return None, "abstention"

        values = {rule[index] for rule in candidates}
        if len(values) == 1:
            return next(iter(values)), "family_agreement"

        # Premiere hypothese compatible : explicitement provisoire.
        return candidates[0][index], "provisional"

    def choose_question(self):
        observations = self.state["observations"]
        known = sorted(map(int, observations))
        unknown = [i for i in range(32) if str(i) not in observations]
        step = len(self.state["events"])

        # Reverifier periodiquement la memoire pour surveiller les changements.
        if known and (step % 4 == 3 or not unknown):
            cursor = self.state.get("audit_cursor", 0)
            index = known[cursor % len(known)]
            self.state["audit_cursor"] = cursor + 1
            return index, "memory_audit"

        candidates = self.candidates()
        if not candidates:
            return unknown[0], "exploration_without_hypothesis"

        def score(index):
            ones = sum(rule[index] for rule in candidates)
            return min(ones, len(candidates)-ones)

        return max(unknown, key=score), "hypothesis_test"

    def observe(self, index, answer, prediction, prediction_kind, purpose):
        observations = self.state["observations"]
        previous = observations.get(str(index))
        changed = previous is not None and previous != answer
        old_epoch = self.state["epoch"]
        old_stage = self.state["stage"]

        if changed:
            # Ne pas effacer l'historique : archiver l'ancien contexte.
            self.state["archives"].append({
                "epoch": old_epoch,
                "stage": old_stage,
                "observations": dict(observations),
                "closed_by": {
                    "input_index": index,
                    "previous_answer": previous,
                    "new_answer": answer,
                },
            })
            self.state["epoch"] += 1
            self.state["audit_cursor"] = 0
            self.state["stage"] = "base"
            self.state["observations"] = {}

        self.state["observations"][str(index)] = answer
        expanded = False

        if not self.candidates() and self.state["stage"] == "base":
            self.state["stage"] = "repair"
            expanded = True

        candidates = self.candidates()
        status = (
            "family_contradicted" if not candidates
            else "fully_observed_epoch"
            if len(self.state["observations"]) == 32
            else "provisional"
        )

        event = {
            "step": len(self.state["events"])+1,
            "epoch_before": old_epoch,
            "epoch_after": self.state["epoch"],
            "input_index": index,
            "prediction": prediction,
            "prediction_kind": prediction_kind,
            "answer": answer,
            "purpose": purpose,
            "prediction_wrong": prediction is not None and prediction != answer,
            "changed_answer_detected": changed,
            "grammar_expanded": expanded,
            "candidate_count": len(candidates),
            "status": status,
        }
        self.state["events"].append(event)
        return event

    def save(self, path):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["format"] != 1:
            raise ValueError("Etat incompatible.")
        return cls(state)

def main():
    output = Path("autonomous_lab/runs/integrated_agent_02")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)
    state_path = output / "agent_state.json"

    # Environnement prive du simulateur : x0 avec une exception.
    first = tuple(
        ((index >> 4) & 1) ^ int(index == 17)
        for index in range(32)
    )
    second = tuple(
        value ^ int(index == 17)
        for index, value in enumerate(first)
    )
    assert first in REPAIR and second in REPAIR

    agent = Agent()
    reloads = 0

    for step in range(96):
        index, purpose = agent.choose_question()
        prediction, kind = agent.prediction(index)

        # Le changement n'est pas transmis a l'agent.
        hidden = first if step < 48 else second
        answer = hidden[index]

        event = agent.observe(index, answer, prediction, kind, purpose)
        agent.save(state_path)

        if event["grammar_expanded"]:
            print(f"Etape {step+1} : extension de la grammaire.", flush=True)
        if event["changed_answer_detected"]:
            print(
                f"Etape {step+1} : reponse modifiee detectee ; "
                f"ancien contexte archive, nouvel episode.",
                flush=True,
            )

        if (step+1) % 16 == 0:
            snapshot = json.dumps(agent.state, sort_keys=True)
            agent = Agent.load(state_path)
            assert json.dumps(agent.state, sort_keys=True) == snapshot
            reloads += 1
            print(
                f"Etape {step+1} : etat recharge | "
                f"epoch={agent.state['epoch']} | "
                f"entrees memorisees={len(agent.state['observations'])}",
                flush=True,
            )

    events = agent.state["events"]
    summary = {
        "interactions": len(events),
        "reloads_checked": reloads,
        "archived_epochs": len(agent.state["archives"]),
        "grammar_expansions": sum(e["grammar_expanded"] for e in events),
        "changed_answers_detected": sum(
            e["changed_answer_detected"] for e in events
        ),
        "predictions_made": sum(e["prediction"] is not None for e in events),
        "incorrect_predictions": sum(e["prediction_wrong"] for e in events),
        "abstentions": sum(e["prediction"] is None for e in events),
        "current_epoch_observed_inputs": len(agent.state["observations"]),
    }

    # Evaluation finale separee : aucun retour a l'agent.
    correct = 0
    answered = 0
    for index in range(32):
        prediction, _ = agent.prediction(index)
        if prediction is not None:
            answered += 1
            correct += prediction == second[index]
    summary["final_evaluator_correct"] = correct
    summary["final_evaluator_answered"] = answered
    summary["final_evaluator_domain"] = 32

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "interactions": 96,
                "environment_switch_after": 48,
                "change_type": "single_output_flip",
                "changed_input_index": 17,
                "audit_policy": "persistent_round_robin_cursor",
                "switch_announced": False,
                "memory_audit_frequency": "every_fourth_interaction",
                "reload_frequency": 16,
                "neural_model_used": False,
                "noise": False,
                "repair_representation": "known_rule_plus_one_exception",
            },
            "summary": summary,
            "events": events,
        }, indent=2),
        encoding="utf-8",
    )
    print("\nBILAN")
    print(json.dumps(summary, indent=2))
    print(f"Resultats : {path}")

if __name__ == "__main__":
    main()
