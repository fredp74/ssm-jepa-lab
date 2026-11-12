import json
import random
import statistics
import time
from pathlib import Path

import torch
from train_rule_ranker import RuleRanker

torch.set_num_threads(1)

def prepare(row):
    observations = row["observations"]
    x = torch.zeros(1, 12, 6)
    for position, item in enumerate(observations):
        x[0, position] = torch.tensor(
            item["input"] + [item["output"]], dtype=torch.float
        )*2-1
    return {
        "observations": observations,
        "x": x,
        "length": torch.tensor([len(observations)]),
        "indices": torch.tensor([
            item["input_index"] for item in observations
        ]),
        "answers": torch.tensor([
            item["output"] for item in observations
        ]),
    }

def scalar_search(case, rules, order):
    for candidate in order:
        if all(
            rules[candidate][item["input_index"]] == item["output"]
            for item in case["observations"]
        ):
            return candidate
    raise RuntimeError("Aucune candidate compatible.")

@torch.inference_mode()
def main():
    data_root = Path("autonomous_lab/data/rule_ranking_01")
    model_root = Path("autonomous_lab/runs/rule_ranking_01")
    output = Path("autonomous_lab/runs/rule_ranking_timing_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")

    protocol = json.loads(
        (data_root / "protocol.json").read_text(encoding="utf-8")
    )
    rules = [tuple(map(int, value)) for value in protocol["rules"]]
    rule_tensor = torch.tensor(rules)
    fixed_order = list(range(len(rules)))

    rows = [
        json.loads(line)
        for line in (data_root / "validation.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    rng = random.Random(92001)
    selected = []
    for count in (4, 8, 12):
        pool = [row for row in rows if len(row["observations"]) == count]
        selected.extend(rng.sample(pool, 100))
    rng.shuffle(selected)
    cases = [prepare(row) for row in selected]

    def fixed(case):
        return scalar_search(case, rules, fixed_order)

    def vectorized(case):
        valid = (
            rule_tensor[:, case["indices"]] == case["answers"]
        ).all(dim=1)
        indices = valid.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            raise RuntimeError("Aucune candidate compatible.")
        return int(indices[0])

    output.mkdir(parents=True)
    results = []

    for seed in (91011, 91012, 91013):
        model = RuleRanker(len(rules))
        model.load_state_dict(torch.load(
            model_root / f"model_{seed}.pt",
            map_location="cpu", weights_only=True,
        ))
        model.eval()

        def guided(case):
            logits = model(case["x"], case["length"])[0]
            order = logits.argsort(
                descending=True, stable=True
            ).tolist()
            return scalar_search(case, rules, order)

        methods = {
            "symbolic_python": fixed,
            "symbolic_vectorized": vectorized,
            "ssm_ranking_and_verification": guided,
        }

        # Verifier les contrats avant de chronometrer.
        for case in cases:
            assert fixed(case) == vectorized(case)
            candidate = guided(case)
            assert all(
                rules[candidate][item["input_index"]] == item["output"]
                for item in case["observations"]
            )

        for case in cases[:15]:
            for function in methods.values():
                function(case)

        measurements = {name: [] for name in methods}
        for repetition in range(5):
            order = list(methods)
            random.Random(92002 + repetition).shuffle(order)
            case_order = list(range(len(cases)))
            random.Random(92010 + repetition).shuffle(case_order)

            for name in order:
                start = time.perf_counter_ns()
                for index in case_order:
                    methods[name](cases[index])
                elapsed = time.perf_counter_ns()-start
                measurements[name].append(elapsed/len(cases)/1000)

        print(f"\nGraine {seed} — microsecondes par episode")
        for name, values in measurements.items():
            median = statistics.median(values)
            print(f"{name} : {median:.2f} us")
            results.append({
                "seed": seed,
                "method": name,
                "microseconds_per_episode": values,
                "median_microseconds": median,
            })

    path = output / "results.json"
    path.write_text(
        json.dumps({
            "config": {
                "episodes": len(cases),
                "episodes_per_observation_count": 100,
                "observation_counts": [4, 8, 12],
                "repetitions": 5,
                "batch_size": 1,
                "cpu_threads": 1,
                "warmup_episodes_per_method": 15,
                "included_in_neural_timing": [
                    "model_forward", "sorting", "candidate_verification"
                ],
                "excluded_from_all_timings": [
                    "file_loading", "model_loading", "input_preparation"
                ],
                "torch_version": str(torch.__version__),
            },
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultats : {path}")

if __name__ == "__main__":
    main()
