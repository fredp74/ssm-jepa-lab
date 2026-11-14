"""Check provenance and split failures before training."""
import json
from pathlib import Path
import tempfile
import unittest
from prepare_finance import prepare, sha

SOURCE = Path(__file__).resolve().parents[1] / "corpora/finance/seed.json"


class FinanceTests(unittest.TestCase):
    def test_manifest_and_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "corpus"
            manifest = prepare(SOURCE, output)
            for split in ("train", "validation"):
                self.assertEqual(
                    sha((output / manifest[split]["file"]).read_bytes()),
                    manifest[split]["sha256"])
            with self.assertRaises(FileExistsError):
                prepare(SOURCE, output)

    def test_reject_bad_records(self):
        for failure in ("group", "duplicate", "rights"):
            records = json.loads(SOURCE.read_text())
            if failure == "group":
                records[-1]["group"] = records[0]["group"]
            elif failure == "duplicate":
                records[-1]["text"] = records[0]["text"]
            else:
                del records[0]["rights_basis"]
            with tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "records.json"
                source.write_text(json.dumps(records), encoding="utf-8")
                with self.assertRaises(ValueError):
                    prepare(source, Path(folder) / "output")
                self.assertFalse((Path(folder) / "output").exists())
