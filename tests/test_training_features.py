import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

import prepare_english as english
import ssm_jepa as system


class TrainingFeatures(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_feedforward_parallel_matches_recurrence(self):
        model = system.LanguageModel(width=16, layers=2, state=3, ffn_mult=2).eval()
        x = torch.randint(256, (2, 70))
        with torch.no_grad():
            parallel = model(x)
            state, values = None, []
            for token in x.unbind(1):
                value, state = model.step(token, state)
                values.append(value)
            self.assertTrue(torch.allclose(parallel, torch.stack(values, 1), atol=2e-5, rtol=2e-5))
            altered = x.clone()
            altered[:, 31:] = (altered[:, 31:] + 1) % 256
            self.assertTrue(torch.equal(parallel[:, :31], model(altered)[:, :31]))

    def corpus(self, directory):
        manifest = {"format": "ssm-jepa-corpus-v1"}
        for split, value in (("train", "A long English training story.\n"),
                             ("validation", "A different English validation story.\n")):
            path = directory / (split + ".txt")
            path.write_text(value*100)
            manifest[split] = {"file": path.name, "sha256": system.digest(path)}
        path = directory / "corpus.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_explicit_split_and_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.corpus(directory)
            data, cut, metadata = system.dataset(path, 64)
            self.assertEqual(cut, (directory / "train.txt").stat().st_size)
            self.assertEqual(len(data)-cut, (directory / "validation.txt").stat().st_size)
            (directory / "validation.txt").write_text("changed")
            with self.assertRaises(ValueError):
                system.dataset(path, 64)

    def test_new_architecture_schedule_exact_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            corpus = self.corpus(directory)
            base = directory / "base.pt"
            system.initialize(argparse.Namespace(
                run=base, from_scratch=True, checkpoint=directory / "missing.pt",
                data=corpus, width=16, layers=2, state_size=3, ffn_mult=2,
                context=64, lr=1e-4, seed=9, warmup=3, decay_steps=10, variance_weight=0.01))
            initial = system.load(base)
            full, resumed = directory / "full.pt", directory / "resumed.pt"
            system.save(full, copy.deepcopy(initial))
            system.save(resumed, copy.deepcopy(initial))
            system.train(argparse.Namespace(run=full, data=None, steps=2))
            system.train(argparse.Namespace(run=resumed, data=None, steps=1))
            system.train(argparse.Namespace(run=resumed, data=None, steps=1))
            a, b = system.load(full), system.load(resumed)
            for part in ("model", "teacher", "predictor"):
                self.assertTrue(all(torch.equal(a[part][key], b[part][key]) for key in a[part]))
            self.assertAlmostEqual(a["optimizer"]["param_groups"][0]["lr"], 1e-4*2/3)
            self.assertTrue(torch.equal(a["generator"], b["generator"]))

    def test_story_boundaries_and_exact_deduplication(self):
        raw = "First story.<|endoftext|>\nSecond story.<|endoftext|>unfinished".encode()
        self.assertEqual(english.stories(raw), ["First story.", "Second story."])
        self.assertEqual(english.unique(["a", "b", "a"], ["b"]), ["a"])
        self.assertEqual(english.stories("é<|endoftext|>".encode() + b"\xc3"), ["é"])


if __name__ == "__main__":
    unittest.main()
