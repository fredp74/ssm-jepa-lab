
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
