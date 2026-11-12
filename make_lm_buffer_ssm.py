from pathlib import Path

src = Path("lm_buffer.py").read_text(encoding="utf-8-sig")

ssm_definition = '''
import functools
import ssm_par
from ssm_par import SSMEnc
ssm_par.pscan = functools.partial(ssm_par.pscan, T=16)

class SLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, 256)
        self.enc = SSMEnc(
            256, 256, depth=6, slow=True, par=True, n=4
        )
        self.head = nn.Linear(256, 256)

    def forward(self, x):
        return self.head(self.enc(self.emb(x)))

factory = SLM
'''

patches = [
    (
        'parser.add_argument("model", choices=["Transformer", "GRU"])',
        'parser.add_argument("model", choices=["SSM-n4"])\n'
        'parser.add_argument("--seeds", type=int, nargs="+", default=[1000])\n'
        'parser.add_argument("--check", action="store_true")'
    ),
    (
        'SEEDS = (1000, 1001, 1002)',
        'SEEDS = tuple(args.seeds)'
    ),
    (
        'CONDS = ("sans replay", "tampon256", "tampon4096", "replay complet")',
        'CONDS = ("sans replay", "tampon4096", "replay complet")'
    ),
    (
        'factory = TLM if args.model == "Transformer" else GLM',
        ssm_definition.strip()
    ),
    (
        'base_a, base_b = bpb(reference, "A"), bpb(reference, "B")',
        '''if args.check:
    torch.manual_seed(123)
    x = torch.randint(0, 256, (1, 37))
    logits = reference(x)
    if logits.shape != (1, 37, 256):
        raise RuntimeError(f"Forme inattendue : {logits.shape}")
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 256),
        x[:, 1:].reshape(-1)
    )
    loss.backward()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Perte non finie")
    for name, parameter in reference.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Gradient absent ou non fini : {name}")
    print(
        "CHECK OK : checkpoint charge, forward et backward finis ; "
        f"{sum(p.numel() for p in reference.parameters())} parametres.",
        flush=True
    )
    raise SystemExit(0)

base_a, base_b = bpb(reference, "A"), bpb(reference, "B")'''
    ),
    (
        'sd = values.std().item() if len(values) > 1 else float("nan")',
        'sd = values.std().item() if len(values) > 1 else float("nan")'
    ),
]

for old, new in patches:
    if src.count(old) != 1:
        raise SystemExit(
            f"Creation interrompue : motif absent ou multiple : {old!r}"
        )
    src = src.replace(old, new, 1)

compile(src, "lm_buffer_ssm.py", "exec")
Path("lm_buffer_ssm.py").write_text(src, encoding="utf-8")
print("Script SSM cree ; syntaxe valide ; original inchange.")
