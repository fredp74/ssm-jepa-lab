import itertools
import json
import random
from pathlib import Path

import torch
from latent_objective import LatentObjective
from torch.nn import functional as F

from meta_ssm import MetaSSM
from meta_ssm import INPUTS, environment

torch.set_num_threads(1)

def rank(vectors):
    basis = {}
    for value in vectors:
        while value:
            pivot = value.bit_length()-1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)

def context_rank(context):
    return rank([
        8 + 4*INPUTS[i][0] + 2*INPUTS[i][1] + INPUTS[i][2]
        for i in context
    ])

def prepare_contexts():
    informative, ambiguous = [], []
    for context in itertools.combinations(range(8), 4):
        bucket = informative if context_rank(context) == 4 else ambiguous
        bucket.append(context)

    assert len(informative) == 56 and len(ambiguous) == 14
    rng = random.Random(610000)
    rng.shuffle(informative)
    rng.shuffle(ambiguous)

    train = informative[:40] + ambiguous[:10]
    test = informative[40:] + ambiguous[10:]
    assert not set(train) & set(test)
    return train, test

@torch.no_grad()
def evaluate(model, contexts):
    model.eval()
    inputs = torch.tensor(INPUTS)
    stats = {
        "correct_unknown": 0,
        "total_unknown": 0,
        "correct_known": 0,
        "total_known": 0,
        "symbolic_score": 0.0,
    }
    details = []

    for context in contexts:
        remaining = [i for i in range(8) if i not in context]
        for rule in range(16):
            ids = torch.tensor([rule])
            x = inputs[list(context)][None]
            y = environment(ids, x)
            queries = inputs[remaining][None]
            expected = environment(ids, queries)
            answer = model(x, y, queries).argmax(-1)
            known = model(x, y, x).argmax(-1)

            correct = int((answer == expected).sum())
            stats["correct_unknown"] += correct
            stats["total_unknown"] += 4
            stats["correct_known"] += int((known == y).sum())
            stats["total_known"] += 4

            # Avec quatre questions, rang 4 : regle identifiee.
            # Sinon, chaque entree restante est ambigue :
            # vote symbolique a egalite, score espere de 0.5.
            stats["symbolic_score"] += (
                4.0 if context_rank(context) == 4 else 2.0
            )
            details.append({
                "context": list(context),
                "rule": rule,
                "rank": context_rank(context),
                "correct": correct,
                "predictions": answer[0].tolist(),
                "expected": expected[0].tolist(),
            })

    return {"summary": stats, "details": details}

def main():
    output = Path("autonomous_lab/runs/context_holdout_latent_01")
    if output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    output.mkdir(parents=True)

    train_contexts, test_contexts = prepare_contexts()
    protocol = {
        "architecture": "diagonal_ssm_with_latent_objective",
        "latent_weight": 0.1,
        "variance_weight": 0.01,
        "teacher_ema": 0.99,
        "rules": list(range(16)),
        "train_contexts": train_contexts,
        "test_contexts": test_contexts,
        "seeds": [61001, 61002, 61003],
        "steps": 1500,
        "batch_size": 32,
        "learning_rate": 0.001,
        "evaluation_updates": 0,
        "training_loss": "unknown_queries_only",
        "torch_version": str(torch.__version__),
    }
    # Enregistrer le partage avant tout apprentissage.
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    inputs = torch.tensor(INPUTS)
    results = []

    for seed in protocol["seeds"]:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed+100)
        model = MetaSSM()
        auxiliary = LatentObjective(model)
        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(auxiliary.predictor.parameters()), lr=0.001
        )

        for step in range(protocol["steps"]):
            rules = torch.randint(16, (32,), generator=generator)
            choices = torch.randint(
                len(train_contexts), (32,), generator=generator
            ).tolist()

            observed_indices = []
            query_indices = []
            for choice in choices:
                context = train_contexts[choice]
                order = torch.randperm(4, generator=generator).tolist()
                observed_indices.append([context[i] for i in order])
                query_indices.append([
                    i for i in range(8) if i not in context
                ])

            x = inputs[torch.tensor(observed_indices)]
            queries = inputs[torch.tensor(query_indices)]
            y = environment(rules, x)
            expected = environment(rules, queries)

            model.train()
            logits = model(x, y, queries)
            loss = F.cross_entropy(
                logits.reshape(-1, 2), expected.reshape(-1)
            )
            latent_loss, variance_loss = auxiliary(model, x, y, rules)
            task_loss = loss
            loss = task_loss + 0.1*latent_loss + 0.01*variance_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Perte non finie.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(auxiliary.predictor.parameters()),
                1.0, error_if_nonfinite=True
            )
            optimizer.step()
            auxiliary.update_teacher(model)

            if (step+1) % 500 == 0:
                print(
                    f"Graine {seed} | {step+1}/1500 | "
                    f"total={loss.item():.4f} | "
                    f"reponse={task_loss.item():.4f} | "
                    f"latent={latent_loss.item():.4f} | "
                    f"variance={variance_loss.item():.4f}", flush=True
                )

        result = {
            "seed": seed,
            "train": evaluate(model, train_contexts),
            "test": evaluate(model, test_contexts),
        }
        results.append(result)
        torch.save(model.state_dict(), output / f"model_{seed}.pt")
        torch.save({
            "auxiliary": auxiliary.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, output / f"training_auxiliary_{seed}.pt")

        for split in ("train", "test"):
            s = result[split]["summary"]
            print(
                f"{split.upper()} | graine {seed} | "
                f"inconnues={s['correct_unknown']}/{s['total_unknown']} | "
                f"memoire={s['correct_known']}/{s['total_known']} | "
                f"reference_symbolique={s['symbolic_score']}/"
                f"{s['total_unknown']}",
                flush=True,
            )

        (output / "results.json").write_text(
            json.dumps({
                "protocol": protocol,
                "results": results,
            }, indent=2),
            encoding="utf-8",
        )

    print(f"Resultats : {output / 'results.json'}")

if __name__ == "__main__":
    main()
