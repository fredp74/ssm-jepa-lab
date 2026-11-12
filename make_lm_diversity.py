from pathlib import Path

src = Path("lm_reservoir.py").read_text(encoding="utf-8-sig")

new_insert = '''    def insert(self, sequences, label):
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
'''

old_insert = '''    def insert(self, sequences, label):
        for sequence in sequences:
            self.seen += 1
            index = torch.randint(
                self.seen, (1,), generator=self.generator
            ).item()
            if index < CAPACITY:
                self.memory[index].copy_(sequence.to(torch.uint8))
                self.labels[index] = label
'''

patches = [
    (
        'ROOT = Path("lm_reservoir_v1")',
        'ROOT = Path("lm_diversity_v1")'
    ),
    (
        '        self.memory = initial.clone().to(torch.uint8)',
        '        self.memory = initial.clone().to(torch.uint8)\n'
        '        self.features = torch.stack([\n'
        '            self.describe(sequence) for sequence in self.memory\n'
        '        ])'
    ),
    (old_insert, new_insert),
    (
        '        self.memory = state["memory"].clone()',
        '        self.memory = state["memory"].clone()\n'
        '        self.features = torch.stack([\n'
        '            self.describe(sequence) for sequence in self.memory\n'
        '        ])'
    ),
]

for old, new in patches:
    if src.count(old) != 1:
        raise SystemExit(
            f"Creation interrompue : motif absent ou multiple : {old!r}"
        )
    src = src.replace(old, new, 1)

compile(src, "lm_diversity.py", "exec")
Path("lm_diversity.py").write_text(src, encoding="utf-8")
print("Script diversite cree ; syntaxe valide ; original inchange.")
