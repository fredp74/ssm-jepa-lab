import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from meta_ssm import MetaSSM, INPUTS, environment
from context_holdout import evaluate

torch.set_num_threads(1)

def main():
    source = Path(
        "autonomous_lab/runs/context_holdout_ssm_01/protocol.json"
    )
    original = json.loads(source.read_text(encoding="utf-8"))
    train_contexts = original["train_contexts"]
    test_contexts = original["test_contexts"]

    output = Path("autonomous_lab/runs/partial_context_ensemble_02")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    seeds = [62101, 62102, 62103]
    config = {
        "architecture": "diagonal_ssm",
        "ensemble_seeds": seeds,
        "train_rules": list(range(16)),
        "observations_per_episode": [0, 1, 2, 3, 4],
        "train_contexts": train_contexts,
        "test_contexts": test_contexts,
        "source_protocol_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "training_steps": 1500,
        "batch_size": 32,
        "learning_rate": 0.001,
        "evaluation_updates": 0,
        "loss": "unknown_queries_only",
        "torch_version": str(torch.__version__),
    }
    (output / "protocol.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    inputs = torch.tensor(INPUTS)
    results = []

    for seed in seeds:
        torch.manual_seed(seed)
        model = MetaSSM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        rule_rng = torch.Generator().manual_seed(seed+100)
        context_rng = torch.Generator().manual_seed(seed+200)

        for step in range(1500):
            count = step % 5
            rules = torch.randint(16, (32,), generator=rule_rng)
            choices = torch.randint(
                len(train_contexts), (32,), generator=context_rng
            ).tolist()

            observed_indices = []
            query_indices = []

            for choice in choices:
                context = train_contexts[choice]
                order = torch.randperm(
                    4, generator=context_rng
                ).tolist()
                observed = [context[i] for i in order[:count]]
                observed_indices.append(observed)
                query_indices.append([
                    i for i in range(8) if i not in observed
                ])

            observed_ids = torch.tensor(
                observed_indices, dtype=torch.long
            ).reshape(32, count)
            query_ids = torch.tensor(query_indices, dtype=torch.long)
            x = inputs[observed_ids]
            queries = inputs[query_ids]
            y = environment(rules, x)
            expected = environment(rules, queries)

            model.train()
            logits = model(x, y, queries)
            loss = F.cross_entropy(
                logits.reshape(-1, 2), expected.reshape(-1)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Perte non finie.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()

            if (step+1) % 500 == 0:
                print(
                    f"Graine {seed} | {step+1}/1500 | "
                    f"observations={count} | perte={loss.item():.4f}",
                    flush=True,
                )

        torch.save(model.state_dict(), output / f"model_{seed}.pt")
        result = {
            "seed": seed,
            "train": evaluate(model, train_contexts),
            "test": evaluate(model, test_contexts),
        }
        results.append(result)

        score = result["test"]["summary"]
        print(
            f"TEST 4 observations | graine {seed} | "
            f"{score['correct_unknown']}/{score['total_unknown']}",
            flush=True,
        )

        (output / "results.json").write_text(
            json.dumps({
                "config": config,
                "results": results,
            }, indent=2),
            encoding="utf-8",
        )

    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
