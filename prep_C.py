import hashlib
import json
import re
import urllib.request
from pathlib import Path

import torch

BOOKS = [
    (4791, "Voyage au centre de la Terre", "train"),
    (5097, "Vingt mille lieues sous les mers", "train"),
    (800, "Le Tour du monde en quatre-vingts jours", "test"),
]

cache = Path("corpus_C_gutenberg")
cache.mkdir(exist_ok=True)
groups = {"train": [], "test": []}
manifest = []

for number, title, split in BOOKS:
    url = f"https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt"
    raw_path = cache / f"pg{number}.txt"

    if raw_path.exists():
        raw = raw_path.read_bytes()
    else:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ssm-jepa-research/1.0"}
        )
        print(f"Telechargement : {title}", flush=True)
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
        # Verifier le decodage avant de conserver le fichier.
        raw.decode("utf-8-sig")
        raw_path.write_bytes(raw)

    text = raw.decode("utf-8-sig")
    start = re.search(
        r"^\s*\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$",
        text, re.MULTILINE | re.IGNORECASE
    )
    end = re.search(
        r"^\s*\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$",
        text, re.MULTILINE | re.IGNORECASE
    )
    if start is None or end is None or start.end() >= end.start():
        raise RuntimeError(
            f"Bornes Gutenberg introuvables pour {number}. "
            "Preparation interrompue."
        )

    body = text[start.end():end.start()]
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    data = (body + "\n").encode("utf-8")
    if len(data) < 100_000:
        raise RuntimeError(f"Texte anormalement court : {title}")

    offset = sum(len(part) for part in groups[split])
    groups[split].append(data)
    manifest.append({
        "id": number,
        "title": title,
        "split": split,
        "url": url,
        "offset": offset,
        "bytes": len(data),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "clean_sha256": hashlib.sha256(data).hexdigest(),
    })
    print(f"{split:5s} | {len(data):9d} octets | {title}", flush=True)

for split, parts in groups.items():
    data = b"".join(parts)
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
    torch.save(tensor, f"dataC_{split}.pt")
    print(f"dataC_{split}.pt : {len(data):,} octets", flush=True)

Path("dataC_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("Preparation terminee ; aucun entrainement lance.")
