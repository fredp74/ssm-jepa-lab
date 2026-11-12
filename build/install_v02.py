import base64, hashlib, json, zlib
from pathlib import Path

if not Path("continual.py").exists():
    raise SystemExit("Lancer depuis ssm_jepa/build.")
raw = zlib.decompress(base64.b64decode(Path("v02.b64").read_text().strip()))
expected = "3cdf89205b2e6e2fbafa1e3a5bce0ed2fb26232dd209041e6fd1c85aff69349a"
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit("Empreinte incorrecte : installation interrompue.")
files = json.loads(raw)
if set(files) != {"assistant_v02.py", "prepare_demo.py"}:
    raise SystemExit("Contenu inattendu.")
for name, code in files.items():
    if Path(name).exists():
        raise SystemExit(f"{name} existe deja : aucun remplacement.")
    compile(code, name, "exec")
for name, code in files.items():
    Path(name).write_text(code, encoding="utf-8")
    print("Installe :", name)
