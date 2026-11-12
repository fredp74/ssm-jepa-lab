from pathlib import Path

jobs = [
    ("lm_reservoir.py", "lm_reservoir_seeds.py"),
    ("lm_reservoir_protected.py", "lm_reservoir_protected_seeds.py"),
]

prepared = []
for original, destination in jobs:
    src = Path(original).read_text(encoding="utf-8-sig")
    old = "SEED = 1000"
    if src.count(old) != 1:
        raise SystemExit(f"Structure inattendue : {original}")

    replacement = '''import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, choices=[1000, 1001, 1002],
                    default=1000)
SEED = parser.parse_args().seed
if SEED != 1000:
    ROOT = ROOT / f"seed_{SEED}"
    ROOT.mkdir(exist_ok=True)
'''
    src = src.replace(old, replacement, 1)
    compile(src, destination, "exec")
    prepared.append((destination, src))

for destination, src in prepared:
    Path(destination).write_text(src, encoding="utf-8")
    print(f"Cree : {destination} ; syntaxe valide")

print("Originaux et resultats seed=1000 inchanges.")
