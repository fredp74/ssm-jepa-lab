import argparse
import copy
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import ssm_jepa as system


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, directory):
        torch.manual_seed(123)
        checkpoint = directory / "initial.pt"
        system.save(checkpoint, system.LanguageModel().state_dict())
        data = directory / "text.txt"
        data.write_text(
            "The model learns from text. A reproducible example.\n" * 100
        )
        run = directory / "base.pt"
        system.initialize(argparse.Namespace(
            run=run, checkpoint=checkpoint, data=data,
            lr=1e-5, seed=123,
        ))
        return run

    def test_recurrent_generation_and_causality(self):
        torch.manual_seed(1)
        model = system.LanguageModel(
            width=16, layers=2, state=2
        ).eval()
        tokens = torch.randint(256, (2, 73))
        with torch.no_grad():
            parallel = model(tokens)
            states = None
            outputs = []
            for token in tokens.unbind(1):
                logits, states = model.step(token, states)
                outputs.append(logits)
            recurrent = torch.stack(outputs, 1)
            self.assertTrue(torch.allclose(
                parallel, recurrent, atol=2e-5, rtol=2e-5
            ))
            changed = tokens.clone()
            changed[:, 35:] = (changed[:, 35:] + 1) % 256
            self.assertTrue(torch.equal(
                parallel[:, :35], model(changed)[:, :35]
            ))

    def test_resume_replay_and_new_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base = self.fixture(directory)
            original = system.load(base)
            one = directory / "one.pt"
            split = directory / "split.pt"
            system.save(one, copy.deepcopy(original))
            system.save(split, copy.deepcopy(original))

            system.train(argparse.Namespace(
                run=one, data=None, steps=2
            ))
            system.train(argparse.Namespace(
                run=split, data=None, steps=1
            ))
            system.train(argparse.Namespace(
                run=split, data=None, steps=1
            ))
            full = system.load(one)
            resumed = system.load(split)
            for key in ("model", "teacher", "predictor"):
                for name in full[key]:
                    self.assertTrue(torch.equal(
                        full[key][name], resumed[key][name]
                    ), (key, name))
            for key in ("generator", "memory"):
                self.assertTrue(torch.equal(full[key], resumed[key]))
            self.assertTrue(torch.equal(
                original["memory"][:128], resumed["memory"][:128]
            ))

            model = system.construct(resumed)[0]
            self.assertIsInstance(system.generate_text(
                model, "The ", 16, 0, 40, 42
            ), str)
            system.evaluate(argparse.Namespace(
                run=split, data=None, samples=3
            ))

            other = directory / "new.txt"
            other.write_text(
                "New observations for continued learning.\n" * 100
            )
            system.train(argparse.Namespace(
                run=split, data=other, steps=1
            ))
            adapted = system.load(split)
            self.assertEqual(adapted["step"], 3)
            self.assertEqual(
                adapted["active_data"]["path"], str(other.resolve())
            )
            self.assertTrue(torch.equal(
                original["memory"][:128], adapted["memory"][:128]
            ))

    def test_interrupt_finishes_step_and_saves(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self.fixture(Path(temporary))
            real_construct = system.construct

            def interrupted_construct(state):
                result = real_construct(state)
                optimizer = result[3]
                original_step = optimizer.step

                def interrupted_step(*args, **kwargs):
                    value = original_step(*args, **kwargs)
                    signal.getsignal(signal.SIGINT)(
                        signal.SIGINT, None
                    )
                    return value

                optimizer.step = interrupted_step
                return result

            with patch.object(
                system, "construct", interrupted_construct
            ):
                system.train(argparse.Namespace(
                    run=run, data=None, steps=5
                ))
            self.assertEqual(system.load(run)["step"], 1)

    def test_initialization_from_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = directory / "text.txt"
            data.write_text(
                "Text for a new model with random weights.\n" * 100
            )
            run = directory / "fresh.pt"
            system.initialize(argparse.Namespace(
                run=run, checkpoint=directory / "absent.pt",
                data=data, lr=1e-4, seed=7, from_scratch=True,
            ))
            self.assertIsNone(
                system.load(run)["initial_checkpoint_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
