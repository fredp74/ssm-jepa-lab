from pathlib import Path

aux_path = Path("autonomous_lab/latent_objective.py")
trainer_path = Path("autonomous_lab/context_holdout_latent.py")
if aux_path.exists() or trainer_path.exists():
    raise SystemExit("Programme deja present : aucun remplacement.")

aux_source = r"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from meta_ssm import INPUTS, environment

def encode(model, x, y):
    h = torch.zeros(x.size(0), model.decay.numel(), device=x.device)
    decay = model.decay.sigmoid()
    for position in range(x.size(1)):
        observation = torch.cat([
            x[:, position].float()*2-1,
            y[:, position, None].float()*2-1,
        ], dim=-1)
        h = decay*h + model.encoder(observation)
    return h

class LatentObjective(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.teacher = copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        width = student.decay.numel()
        self.predictor = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
        )

    def forward(self, student, x, y, rules):
        h = encode(student, x, y)
        prediction = self.predictor(h)

        with torch.no_grad():
            full_x = torch.tensor(
                INPUTS, dtype=torch.long, device=x.device
            )[None].expand(x.size(0), -1, -1)
            full_y = environment(rules, full_x)
            target = encode(self.teacher, full_x, full_y)

        latent_loss = F.mse_loss(
            F.layer_norm(prediction, (prediction.size(-1),)),
            F.layer_norm(target, (target.size(-1),)),
        )
        std = torch.sqrt(h.var(dim=0, unbiased=False) + 0.0001)
        variance_loss = F.relu(1.0-std).mean()
        return latent_loss, variance_loss

    @torch.no_grad()
    def update_teacher(self, student):
        for target, source in zip(
            self.teacher.parameters(), student.parameters()
        ):
            target.mul_(0.99).add_(source, alpha=0.01)
"""

source = Path("autonomous_lab/context_holdout_ssm.py").read_text(
    encoding="utf-8-sig"
)

patches = [
    (
        "import torch",
        "import torch\nfrom latent_objective import LatentObjective",
    ),
    (
        'Path("autonomous_lab/runs/context_holdout_ssm_01")',
        'Path("autonomous_lab/runs/context_holdout_latent_01")',
    ),
    (
        '"architecture": "diagonal_ssm"',
        '"architecture": "diagonal_ssm_with_latent_objective",\n'
        '        "latent_weight": 0.1,\n'
        '        "variance_weight": 0.01,\n'
        '        "teacher_ema": 0.99',
    ),
    (
        "        model = MetaSSM()",
        "        model = MetaSSM()\n"
        "        auxiliary = LatentObjective(model)",
    ),
    (
        "torch.optim.Adam(model.parameters(), lr=0.001)",
        "torch.optim.Adam(\n"
        "            list(model.parameters())\n"
        "            + list(auxiliary.predictor.parameters()), lr=0.001\n"
        "        )",
    ),
    (
        "            if not torch.isfinite(loss):",
        "            latent_loss, variance_loss = auxiliary(model, x, y, rules)\n"
        "            task_loss = loss\n"
        "            loss = task_loss + 0.1*latent_loss + 0.01*variance_loss\n"
        "            if not torch.isfinite(loss):",
    ),
    (
        "                model.parameters(), 1.0, error_if_nonfinite=True",
        "                list(model.parameters())\n"
        "                + list(auxiliary.predictor.parameters()),\n"
        "                1.0, error_if_nonfinite=True",
    ),
    (
        "            optimizer.step()",
        "            optimizer.step()\n"
        "            auxiliary.update_teacher(model)",
    ),
    (
        'f"perte={loss.item():.4f}", flush=True',
        'f"total={loss.item():.4f} | "\n'
        '                    f"reponse={task_loss.item():.4f} | "\n'
        '                    f"latent={latent_loss.item():.4f} | "\n'
        '                    f"variance={variance_loss.item():.4f}", flush=True',
    ),
    (
        '        torch.save(model.state_dict(), output / f"model_{seed}.pt")',
        '        torch.save(model.state_dict(), output / f"model_{seed}.pt")\n'
        '        torch.save({\n'
        '            "auxiliary": auxiliary.state_dict(),\n'
        '            "optimizer": optimizer.state_dict(),\n'
        '        }, output / f"training_auxiliary_{seed}.pt")',
    ),
]

for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

compile(aux_source, str(aux_path), "exec")
compile(source, str(trainer_path), "exec")
aux_path.write_text(aux_source, encoding="utf-8")
trainer_path.write_text(source, encoding="utf-8")
print("Deux programmes crees.")
