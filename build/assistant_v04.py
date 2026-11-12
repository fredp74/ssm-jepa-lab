import argparse
import sys
from pathlib import Path

import torch
from torch import nn
import assistant_v03 as previous

core = previous.core
ORIGINAL_FACTORY = core.model_for
ORIGINAL_STREAM = previous.stream_bytes
ARCH = "GRU-attention-v1"


class Hybrid(core.GRU):
    def __init__(self, window=512):
        super().__init__()
        self.window = window
        self.attention = nn.MultiheadAttention(
            384, 4, dropout=0.0, batch_first=True
        )
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, x):
        recurrent, _ = self.g(self.emb(x))
        h = self.norm(recurrent)
        indices = torch.arange(x.size(1), device=x.device)
        distance = indices[:, None]-indices[None, :]
        mask = (distance < 0) | (distance >= self.window)
        attended, _ = self.attention(
            h, h, h, attn_mask=mask, need_weights=False
        )
        return self.head(h+attended)


def factory(state):
    if state["arch"] != ARCH:
        return ORIGINAL_FACTORY(state)
    model = Hybrid(state["attention_window"])
    model.load_state_dict(state["model"])
    return model


def instruction_model(state):
    if state["arch"] not in ("GRU", ARCH):
        raise ValueError("Instructions v0.4 : GRU et GRU-attention uniquement")
    return factory(state)


@torch.no_grad()
def stream(model, prompt, count, temperature, topk, seed, stop=None):
    if not isinstance(model, Hybrid):
        return ORIGINAL_STREAM(
            model, prompt, count, temperature, topk, seed, stop
        )
    if not prompt:
        raise ValueError("Prefixe vide")
    model.eval()
    hidden = None
    cache = None

    def consume(data, hidden, cache):
        x = torch.tensor([list(data)], dtype=torch.long)
        recurrent, hidden = model.g(model.emb(x), hidden)
        h = model.norm(recurrent)
        cache = h if cache is None else torch.cat([cache, h], dim=1)
        cache = cache[:, -model.window:]
        query = h[:, -1:]
        attended, _ = model.attention(
            query, cache, cache, need_weights=False
        )
        logits = model.head(query+attended)[0, 0]
        return logits, hidden, cache

    for start in range(0, len(prompt), 256):
        logits, hidden, cache = consume(
            prompt[start:start+256], hidden, cache
        )

    generator = torch.Generator().manual_seed(seed)
    result = bytearray()
    for _ in range(count):
        if temperature == 0:
            byte = logits.argmax().item()
        else:
            values, indices = (logits/temperature).topk(topk)
            index = torch.multinomial(
                values.softmax(0), 1, generator=generator
            )
            byte = indices[index].item()
        result.append(byte)
        if stop and result.endswith(stop):
            return bytes(result[:-len(stop)]), True
        logits, hidden, cache = consume(
            bytes([byte]), hidden, cache
        )
    return bytes(result), False


def initialize(args):
    if args.run.exists():
        raise FileExistsError("Destination existante : choisir un autre nom")
    state = core.load(args.base)
    if state["arch"] != "GRU":
        raise ValueError("Partir du checkpoint GRU apres C")
    phase = state.get("phase")
    if phase and phase["step"] < phase["steps"]:
        raise ValueError("Phase LM encore en cours")
    if state.get("sft"):
        raise ValueError("Utiliser le GRU apres C avant les instructions")

    torch.manual_seed(1000)
    original = ORIGINAL_FACTORY(state).eval()
    model = Hybrid(args.window).eval()
    missing, unexpected = model.load_state_dict(
        state["model"], strict=False
    )
    expected = {
        "attention.in_proj_weight",
        "attention.in_proj_bias",
        "attention.out_proj.weight",
        "attention.out_proj.bias",
    }
    if unexpected or set(missing) != expected:
        raise ValueError("Parametres du checkpoint incompatibles")

    with torch.no_grad():
        x = torch.arange(37).unsqueeze(0)
        torch.testing.assert_close(
            model(x), original(x), rtol=0, atol=0
        )

    state["arch"] = ARCH
    state["attention_window"] = args.window
    state["model"] = model.state_dict()
    state["optimizer"] = None
    state["torch_rng"] = torch.get_rng_state()
    state["adapter_source_sha256"] = core.fingerprint(args.base)
    core.atomic_save(state, args.run)
    count = sum(p.numel() for p in model.attention.parameters())
    print(
        f"Initialise : {args.run} | sorties initiales identiques "
        f"au GRU | +{count} parametres"
    )


def smoke():
    previous.smoke()
    torch.manual_seed(41)
    base = core.GRU().eval()
    model = Hybrid(window=16).eval()
    model.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(256, (1, 43))

    with torch.no_grad():
        torch.testing.assert_close(
            model(x), base(x), rtol=0, atol=0
        )
        nn.init.normal_(
            model.attention.out_proj.weight, std=0.01
        )
        full = model(x)
        changed = x.clone()
        changed[:, 20:] = torch.randint(
            256, changed[:, 20:].shape
        )
        torch.testing.assert_close(
            full[:, :20], model(changed)[:, :20],
            rtol=1e-4, atol=3e-6
        )

        hidden, cache, pieces = None, None, []
        for position in range(x.size(1)):
            recurrent, hidden = model.g(
                model.emb(x[:, position:position+1]), hidden
            )
            h = model.norm(recurrent)
            cache = h if cache is None else torch.cat([cache, h], 1)
            cache = cache[:, -model.window:]
            attended, _ = model.attention(
                h, cache, cache, need_weights=False
            )
            pieces.append(model.head(h+attended))
        torch.testing.assert_close(
            full, torch.cat(pieces, 1),
            rtol=1e-4, atol=3e-6
        )

        prompt = bytes(x[0].tolist())
        generated, _ = stream(model, prompt, 5, 0, 40, 42)
        expected = bytearray()
        all_ids = x.clone()
        for _ in range(5):
            byte = model(all_ids)[0, -1].argmax().item()
            expected.append(byte)
            all_ids = torch.cat([
                all_ids, torch.tensor([[byte]])
            ], 1)
        assert generated == bytes(expected)

    initial = Hybrid(window=16)
    loss = initial(x).square().mean()
    loss.backward()
    grad = initial.attention.out_proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().max() > 0

    print(
        "HYBRID SMOKE OK : initialisation identique, causalite, "
        "fenetre glissante, generation equivalente et gradient adaptateur."
    )


core.model_for = factory
previous.gru_model = instruction_model
previous.stream_bytes = stream


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "smoke":
        smoke()
    elif command == "init-hybrid":
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--base", type=Path, required=True)
        parser.add_argument("--run", type=Path, required=True)
        parser.add_argument("--window", type=int, default=512)
        args = parser.parse_args()
        if args.window < 1:
            parser.error("window doit etre positif")
        initialize(args)
    elif command == "eval-lm":
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--run", type=Path, required=True)
        parser.add_argument("--samples", type=int, default=200)
        args = parser.parse_args()
        if args.samples < 1:
            parser.error("samples doit etre positif")
        core.evaluate(args)
    else:
        previous.main()


if __name__ == "__main__":
    main()
