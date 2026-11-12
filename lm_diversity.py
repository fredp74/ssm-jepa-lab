from pathlib import Path
import csv

source = Path("lm_abc.py").read_text(encoding="utf-8-sig")
marker = "\nbase_state = load("
if source.count(marker) != 1:
    raise RuntimeError("Structure lm_abc.py inattendue")
prefix = source.split(marker, 1)[0]
old = 'ROOT = Path("lm_abc_v1")'
if prefix.count(old) != 1:
    raise RuntimeError("Repertoire inattendu")
prefix = prefix.replace(old, 'ROOT = Path("lm_diversity_v1")', 1)
exec(compile(prefix, "abc_setup", "exec"))

SEED = 1000
CAPACITY = 4096

class Reservoir:
    def __init__(self, initial, seed):
        self.memory = initial.clone().to(torch.uint8)
        self.features = torch.stack([
            self.describe(sequence) for sequence in self.memory
        ])
        self.seen = len(initial)
        self.generator = torch.Generator().manual_seed(seed)
        # Etiquettes uniquement pour auditer la composition.
        # Elles ne participent jamais a la selection.
        self.labels = torch.zeros(CAPACITY, dtype=torch.long)

    def sample(self, n, generator):
        indices = torch.randint(CAPACITY, (n,), generator=generator)
        return self.memory[indices].long()

    def insert(self, sequences, label):
        for sequence in sequences:
            self.seen += 1
            feature = self.describe(sequence)

            # 32 souvenirs distincts, echantillonnes sans biais de domaine.
            indices = torch.randint(
                CAPACITY, (64,), generator=self.generator
            ).unique()
            while len(indices) < 32:
                extra = torch.randint(
                    CAPACITY, (32,), generator=self.generator
                )
                indices = torch.cat([indices, extra]).unique()
            indices = indices[
                torch.randperm(len(indices), generator=self.generator)[:32]
            ]

            features = torch.cat([
                self.features[indices], feature.unsqueeze(0)
            ])
            # Distance de Hellinger, a un facteur constant pres.
            distances = torch.cdist(
                features, features,
                compute_mode="donot_use_mm_for_euclid_dist"
            )
            distances = distances.masked_fill(
                torch.tril(torch.ones_like(distances, dtype=torch.bool)),
                float("inf")
            )
            flat = distances.argmin().item()
            left, right = divmod(flat, len(features))

            # La nouvelle fenetre est redondante avec un souvenir.
            if right == 32:
                continue

            chosen = left if torch.randint(
                2, (1,), generator=self.generator
            ).item() == 0 else right
            index = indices[chosen].item()
            self.memory[index].copy_(sequence.to(torch.uint8))
            self.features[index].copy_(feature)
            self.labels[index] = label

    @staticmethod
    def describe(sequence):
        counts = torch.bincount(
            sequence.long(), minlength=256
        ).float()
        return (counts / counts.sum()).sqrt()

    def state(self):
        return {
            "memory": self.memory,
            "labels": self.labels,
            "seen": self.seen,
            "rng": self.generator.get_state(),
        }

    def restore(self, state):
        self.memory = state["memory"].clone()
        self.features = torch.stack([
            self.describe(sequence) for sequence in self.memory
        ])
        self.labels = state["labels"].clone()
        self.seen = int(state["seen"])
        self.generator.set_state(state["rng"])

    def composition(self):
        counts = torch.bincount(self.labels, minlength=3).tolist()
        return dict(zip("ABC", counts))

def save_bundle(path, model, reservoir, minutes):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "reservoir": reservoir.state(),
        "training_minutes": minutes,
    }, temporary)
    temporary.replace(path)

def phase(model, reservoir, domain, label):
    model.train()
    main = torch.Generator().manual_seed(SEED)
    replay_rng = torch.Generator().manual_seed(SEED+100_000)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )
    start = time.perf_counter()

    for step in range(STEPS):
        lr = LR*min(1.0, (step+1)/WARM)*(
            0.1+0.45*(1+math.cos(math.pi*step/STEPS))
        )
        for group in opt.param_groups:
            group["lr"] = lr

        incoming = (
            batch_ab("B", B, main) if domain == "B"
            else batch_c(B, main)
        )
        replay = reservoir.sample(BR, replay_rng)
        x = torch.cat([incoming, replay])
        loss = F.cross_entropy(
            model(x[:, :-1]).reshape(-1, 256),
            x[:, 1:].reshape(-1)
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError("Perte non finie")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Les nouvelles sequences entrent apres le replay et l'update.
        reservoir.insert(incoming, label)

        if (step+1) % 500 == 0:
            print(
                f"{domain} pas={step+1} | "
                f"memoire={reservoir.composition()} | "
                f"{(time.perf_counter()-start)/60:.1f} min",
                flush=True
            )
    return (time.perf_counter()-start)/60

torch.manual_seed(SEED)
model = new_model(load("lm_Transformer_A.pt"))
reservoir = Reservoir(pool_a, SEED+300_000)
rows = []

for domain, label in (("B", 1), ("C", 2)):
    path = ROOT / f"Transformer_{SEED}_{domain}.pt"
    if path.exists():
        saved = load(path)
        model.load_state_dict(saved["model"])
        reservoir.restore(saved["reservoir"])
        minutes = saved["training_minutes"]
        print(f"Reprise checkpoint apres {domain}", flush=True)
    else:
        minutes = phase(model, reservoir, domain, label)
        save_bundle(path, model, reservoir, minutes)

    scores = evaluate(model)
    counts = reservoir.composition()
    row = {
        "seed": SEED, "phase": domain,
        **scores,
        **{f"memoire_{k}": counts[k] for k in "ABC"},
        "training_minutes": minutes,
    }
    rows.append(row)
    with (ROOT / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"RESULTAT apres {domain} | "
        f"A {scores['A']:.3f} | B {scores['B']:.3f} | "
        f"C {scores['C']:.3f} | memoire {counts}",
        flush=True
    )

print(f"Resultats : {ROOT / 'results.csv'}")
