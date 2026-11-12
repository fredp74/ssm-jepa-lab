import base64, hashlib, zlib
from pathlib import Path

root = Path("build")
destination = root / "continual.py"
if destination.exists():
    raise SystemExit("continual.py existe deja : aucun remplacement effectue.")
code = zlib.decompress(base64.b64decode(
    (root / "continual.b64").read_text("ascii").strip()
))
expected = "0783008c454df66cc6f7090e52ef5cd12d9d496115d6c810b3f81fe17d2ccc0e"
if hashlib.sha256(code).hexdigest() != expected:
    raise SystemExit("Empreinte incorrecte : installation interrompue.")
compile(code, str(destination), "exec")
destination.write_bytes(code)
print("Build installe ; empreinte et syntaxe valides.")
