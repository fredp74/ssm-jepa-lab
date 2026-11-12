import os
p = os.path.expanduser("~/ssm_jepa/seeds.py")
s = open(p, encoding="utf-8-sig").read()
gru = '''class GRUEnc(nn.Module):
    def __init__(s, din, d):
        super().__init__()
        s.p = nn.Linear(din, d)
        s.g = nn.GRU(d, d, num_layers=2, batch_first=True)
        s.norm = nn.LayerNorm(d)

    def forward(s, x):
        return s.norm(s.g(s.p(x))[0])

'''
assert "def make_batch" in s and "enc = Encoder(C, D)" in s
s = s.replace("def make_batch", gru + "def make_batch", 1)
s = s.replace("enc = Encoder(C, D)", "enc = GRUEnc(C, D)")
s = s.replace('res = {"apres A": []}', 'print("params SSM", sum(q.numel() for q in Encoder(C, D).parameters()), "| GRU", sum(q.numel() for q in GRUEnc(C, D).parameters()))\nres = {"apres A": []}', 1)
open(os.path.expanduser("~/ssm_jepa/seeds_gru.py"), "w", encoding="utf-8").write(s)
