from pathlib import Path

model_path = Path("autonomous_lab/keyed_memory.py")
trainer_path = Path("autonomous_lab/train_keyed_memory.py")
if model_path.exists() or trainer_path.exists():
    raise SystemExit("Programme deja present : aucun remplacement.")

model_source = r"""
import torch
from torch import nn

class KeyedMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(19, 221),
            nn.Tanh(),
            nn.Linear(221, 2),
        )

    def forward(self, observed_x, observed_y, queries):
        batch = queries.size(0)
        known = torch.zeros(batch, 8, device=queries.device)
        values = torch.zeros(batch, 8, device=queries.device)
        batch_ids = torch.arange(batch, device=queries.device)

        # Adressage explicite des huit entrees binaires.
        for position in range(observed_x.size(1)):
            x = observed_x[:, position]
            address = 4*x[:, 0] + 2*x[:, 1] + x[:, 2]
            known[batch_ids, address] = 1
            values[batch_ids, address] = (
                observed_y[:, position].float()*2 - 1
            )

        memory = torch.cat([known, values], dim=-1)
        memory = memory[:, None].expand(-1, queries.size(1), -1)
        logits = self.decoder(torch.cat([
            memory, queries.float()*2 - 1
        ], dim=-1))

        addresses = (
            4*queries[:, :, 0]
            + 2*queries[:, :, 1]
            + queries[:, :, 2]
        )
        is_known = known.gather(1, addresses).bool()
        remembered = (values.gather(1, addresses) + 1)/2

        exact_logits = torch.stack([
            20*(1-remembered),
            20*remembered,
        ], dim=-1)

        return torch.where(is_known[:, :, None], exact_logits, logits)
"""

source = Path(
    "autonomous_lab/compare_observation_curriculum.py"
).read_text(encoding="utf-8-sig")

patches = [
    (
        "from meta_ssm import MetaSSM, INPUTS, QUESTIONS, environment",
        "from meta_ssm import INPUTS, QUESTIONS, environment\n"
        "from keyed_memory import KeyedMemory",
    ),
    (
        'Path("autonomous_lab/runs/observation_curriculum_01")',
        'Path("autonomous_lab/runs/keyed_memory_01")',
    ),
    (
        'for arm in ("fixed", "variable"):',
        'for arm in ("keyed_memory",):',
    ),
    (
        "            model = MetaSSM()",
        "            model = KeyedMemory()",
    ),
    (
        '                if arm == "fixed":',
        "                if False:  # Protocole variable uniquement.",
    ),
    (
        '                        "train_rules": TRAIN_RULES,',
        '                        "architecture": "exact_keyed_memory_plus_mlp",\n'
        '                        "parameters": 4864,\n'
        '                        "known_queries": "exact_lookup",\n'
        '                        "train_rules": TRAIN_RULES,',
    ),
]

for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

compile(model_source, str(model_path), "exec")
compile(source, str(trainer_path), "exec")
model_path.write_text(model_source, encoding="utf-8")
trainer_path.write_text(source, encoding="utf-8")
print("Deux programmes crees.")
