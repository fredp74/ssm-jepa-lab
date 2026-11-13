"""Prepare a bounded, attributed English TinyStories corpus."""
import argparse
import codecs
import hashlib
import json
import math
import shutil
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

SOURCE = "https://huggingface.co/datasets/roneneldan/TinyStories"


def fetch(url, limit):
    request = urllib.request.Request(
        url, headers={"User-Agent": "ssm-jepa-corpus/1.0", "Range": f"bytes=0-{limit-1}"}
    )
    chunks = []
    with urllib.request.urlopen(request, timeout=90) as response:
        remaining = limit
        while remaining:
            chunk = response.read(min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    return b"".join(chunks)


def stories(raw):
    # Ignore only an incomplete final UTF-8 sequence and final story.
    text = codecs.getincrementaldecoder("utf-8")().decode(raw, final=False)
    pieces = text.split("<|endoftext|>")[:-1]
    return [unicodedata.normalize("NFC", s).strip() for s in pieces if s.strip()]


def unique(items, excluded=None):
    seen = set() if excluded is None else set(excluded)
    result = []
    for story in items:
        if story not in seen:
            seen.add(story)
            result.append(story)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/english"))
    parser.add_argument("--train-mib", type=float, default=16)
    parser.add_argument("--validation-mib", type=float, default=1)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.train_mib, args.validation_mib)):
        parser.error("Volumes positifs attendus.")
    if args.output.exists():
        raise SystemExit("Destination existante : aucun remplacement.")
    with urllib.request.urlopen(
        "https://huggingface.co/api/datasets/roneneldan/TinyStories", timeout=90
    ) as response:
        revision = json.load(response)["sha"]
    provenance = {
        "dataset": SOURCE, "revision": revision, "language": "en",
        "license": "cdla-sharing-1.0", "license_url":
        "https://cdla.dev/sharing-1-0/", "sampling": "prefix_with_complete_stories",
        "normalization": "NFC; trim stories; remove exact duplicates and split overlap",
    }
    prepared = {}
    train_stories = []
    for split, name, mib in (
        ("train", "TinyStories-train.txt", args.train_mib),
        ("validation", "TinyStories-valid.txt", args.validation_mib),
    ):
        url = f"{SOURCE}/resolve/{revision}/{name}?download=true"
        raw = fetch(url, int(mib*1048576))
        items = unique(stories(raw), train_stories if split == "validation" else None)
        encoded = ("\n\n".join(items) + "\n").encode("utf-8")
        if len(encoded) < 1285:
            raise ValueError(f"Pas assez de texte complet : {split}")
        if split == "train":
            train_stories = items
        prepared[split] = (encoded, {
            "file": f"{split}.txt", "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded), "stories": len(items), "source_url": url,
            "downloaded_prefix_bytes": len(raw),
        })
        print(f"{split} : {len(items)} histoires, {len(encoded)/1048576:.2f} Mo", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".english-", dir=args.output.parent))
    try:
        manifest = {"format": "ssm-jepa-corpus-v1", "provenance": provenance}
        for split, (encoded, metadata) in prepared.items():
            (staging / metadata["file"]).write_bytes(encoded)
            manifest[split] = metadata
        (staging / "corpus.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if args.output.exists():
            raise FileExistsError(args.output)
        staging.rename(args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Corpus : {args.output / 'corpus.json'}")


if __name__ == "__main__":
    main()
