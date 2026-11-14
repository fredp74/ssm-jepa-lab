"""Build a provenance-tracked financial corpus without changing model weights."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unicodedata


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}")
    records = json.loads(source.read_text(encoding="utf-8"))
    groups, ids, texts = {}, set(), set()
    partitions = {"train": [], "validation": []}
    provenance = []
    for record in records:
        for field in ("id", "group", "split", "title", "published", "period", "units",
                      "rights", "rights_basis", "references", "text", "kind"):
            if not record.get(field):
                raise ValueError(f"Missing field: {field}")
        date.fromisoformat(record["published"])
        split = record["split"]
        if split not in partitions or record["id"] in ids:
            raise ValueError("Invalid split or duplicate document ID")
        ids.add(record["id"])
        if groups.setdefault(record["group"], split) != split:
            raise ValueError("Document group crosses partitions")
        text = unicodedata.normalize("NFC", record["text"]).strip()
        fingerprint = sha(text.encode("utf-8"))
        if fingerprint in texts:
            raise ValueError("Duplicate document text")
        texts.add(fingerprint)
        header = (f"Title: {record['title']}\nPublished: {record['published']}\n"
                  f"Period: {record['period']}\nUnits: {record['units']}\n"
                  f"Document ID: {record['id']}\nKind: {record['kind']}\n")
        partitions[split].append(header + "\n" + text)
        provenance.append({k: v for k, v in record.items() if k != "text"} |
                          {"normalized_text_sha256": fingerprint})
    prepared = {}
    for split, documents in partitions.items():
        raw = ("\n\n".join(documents) + "\n").encode("utf-8")
        if len(raw) < 1285:
            raise ValueError(f"Partition too short: {split}")
        prepared[split] = raw
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".finance-", dir=output.parent))
    try:
        manifest = {"format": "ssm-jepa-corpus-v1", "provenance": {
            "language": "en", "source_registry_sha256": sha(source.read_bytes()),
            "documents": provenance, "normalization": "NFC, strip document edges",
            "split_policy": "Explicit document groups; not a temporal market evaluation",
            "purpose": "Small educational seed; not investment evidence"}}
        for split, raw in prepared.items():
            filename = split + ".txt"
            (staging / filename).write_bytes(raw)
            manifest[split] = {"file": filename, "sha256": sha(raw), "bytes": len(raw),
                               "documents": len(partitions[split])}
        (staging / "corpus.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "corpora/finance/seed.json")
    parser.add_argument("--output", type=Path, default=Path("data/finance_seed"))
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    for split in ("train", "validation"):
        print(split, manifest[split])
    print("Corpus:", args.output / "corpus.json")


if __name__ == "__main__":
    main()
