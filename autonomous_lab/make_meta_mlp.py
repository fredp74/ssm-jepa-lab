from pathlib import Path

source_path = Path("autonomous_lab/meta_ssm.py")
destination = Path("autonomous_lab/meta_mlp.py")
if destination.exists():
    raise SystemExit("Programme deja present : aucun remplacement.")

source = source_path.read_text(encoding="utf-8-sig")

model_code = '''
class MetaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(19, 221),
            nn.Tanh(),
            nn.Linear(221, 2),
        )

    def forward(self, observed_x, observed_y, queries):
        batch = queries.size(0)
        count = observed_x.size(1)
        if count > 4:
            raise ValueError("Quatre observations maximum.")

        observations = torch.cat([
            observed_x.float() * 2 - 1,
            observed_y[:, :, None].float() * 2 - 1,
        ], dim=-1)

        # Les emplacements sans observation sont remplis de zeros.
        memory = queries.new_zeros((batch, 4, 4)).float()
        memory[:, :count] = observations
        flat = memory.reshape(batch, 16)
        flat = flat[:, None, :].expand(-1, queries.size(1), -1)

        inputs = torch.cat([
            flat, queries.float() * 2 - 1
        ], dim=-1)
        return self.network(inputs)

'''

patches = [
    (
        'Path("autonomous_lab/runs/meta_ssm_01")',
        'Path("autonomous_lab/runs/meta_mlp_01")',
    ),
    ("def main():", model_code + "\ndef main():"),
    ("        model = MetaSSM()", "        model = MetaMLP()"),
    (
        "    config = {",
        '    config = {\n'
        '        "architecture": "mlp_flattened_observations",\n'
        '        "parameters": sum(p.numel() for p in MetaMLP().parameters()),',
    ),
]

for old, new in patches:
    if source.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {old!r}")
    source = source.replace(old, new, 1)

compile(source, str(destination), "exec")
destination.write_text(source, encoding="utf-8")
print(f"Programme cree : {destination}")
