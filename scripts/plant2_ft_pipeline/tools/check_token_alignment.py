#!/usr/bin/env python3
"""Show which sequence positions the forecasting logits are taken from.

HFLM.forward appends `speed_token` to the END of the sequence but increments
`remove_idxs` -- the count of tokens at the FRONT -- and then slices
`logits = x[:, remove_idxs:]`. If that increment is wrong, object token j+1 is
supervised with object j's target and the trailing speed token (the one the
ego-speed head reads) is supervised with the last object's box.

This is decided structurally, not statistically: the BERT trunk is replaced by
a stub whose hidden state at position i is the constant i, and the forecasting
heads by a channel mean. Whatever the logits print IS the position each target
was matched against.

  python3 check_token_alignment.py                  # needs PLAN_T or a repo checkout
  PLAN_T=/path/to/plant2/PlanT python3 check_token_alignment.py

Expected on the unfixed code (n_objects=4):
  logits row 0 <- position 33, but target row 0 is object 0 at position 32
  last logits row <- the speed token
Expected after the fix: rows map to positions 32..35 and the speed token is out.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml


def _plant_dir() -> Path:
    env = os.environ.get("PLAN_T")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "plant2" / "PlanT"
        if (cand / "model.py").is_file():
            return cand
    raise SystemExit("cannot locate plant2/PlanT -- set PLAN_T=/path/to/plant2/PlanT")


class DictAsMember(dict):
    """Same attribute-access wrapper dataset.py uses for its standalone run."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return DictAsMember(value) if isinstance(value, dict) else value


class PositionStub(torch.nn.Module):
    """Stands in for the BERT trunk: hidden state at position i is i."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.embeddings = DictAsMember({})

    def forward(self, inputs_embeds=None, **_):
        b, seq, _ = inputs_embeds.shape
        pos = torch.arange(seq, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        hidden = pos[None, :, None].expand(b, seq, self.n_embd).contiguous()

        class Out:
            pass

        out = Out()
        out.last_hidden_state = hidden
        out.attentions = None
        return out


class MeanHead(torch.nn.Module):
    """Stands in for a forecasting head: passes the position through."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab

    def forward(self, x):
        return x.mean(dim=-1, keepdim=True).expand(*x.shape[:-1], self.vocab)


def _stub_transformers_if_missing() -> bool:
    """The trunk is replaced by PositionStub anyway, so a missing `transformers`
    must not stop the check -- it lets this run on a laptop, off the cluster."""
    try:
        import transformers  # noqa: F401
        return False
    except ImportError:
        pass

    import types

    class _Cfg:
        hidden_size = 512  # bert-medium

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = types.SimpleNamespace(word_embeddings=None)
            self.pooler = None

    class _Auto:
        @staticmethod
        def from_pretrained(_name):
            return _Cfg()

        @staticmethod
        def from_config(config=None):
            return _Model()

    mod = types.ModuleType("transformers")
    mod.AutoConfig = _Auto
    mod.AutoModel = _Auto
    mod.BertConfig = _Auto
    sys.modules["transformers"] = mod
    return True


def build_model(plant: Path):
    sys.path.insert(0, str(plant))
    cfg = yaml.safe_load((plant / "config" / "config.yaml").read_text())
    cfg["model"] = yaml.safe_load((plant / "config" / "model" / "PlanT.yaml").read_text())
    # resnet18 weights would be fetched from the network; the trunk is stubbed
    # out anyway, so the BEV branch is not needed to answer the question.
    cfg["model"]["training"]["input_bev"] = False
    cfg["visualize"] = False
    cfg = DictAsMember(cfg)

    from model import HFLM  # noqa: E402  (needs sys.path above)

    model = HFLM(cfg.model.network, cfg)
    model.model = PositionStub(model.n_embd)
    model.heads = torch.nn.ModuleList([MeanHead(v) for v in model.vocab_size])
    model.eval()
    return model, cfg


def make_batch(n_objects: int, n_embd_unused: int = 0):
    """One sample, `n_objects` objects -- the shape generate_batch produces."""
    pool = [[0.0] * 7] + [[1.0, float(i), 0.0, 0.0, 0.0, 4.0, 2.0] for i in range(n_objects)]
    return {
        "idxs": torch.arange(1, n_objects + 1, dtype=torch.int32)[None, :],
        "x_objs": torch.tensor(pool, dtype=torch.float32),
        "route_original": torch.zeros(1, 20, 2),
        "speed_limit": torch.zeros(1, dtype=torch.long),
        "sign_id": torch.tensor([6], dtype=torch.long),
        "y_objs": torch.zeros(n_objects + 1, 4, dtype=torch.long),
    }


def main() -> None:
    plant = _plant_dir()
    stubbed = _stub_transformers_if_missing()
    model, cfg = build_model(plant)

    n_objects = 4
    batch = make_batch(n_objects)
    with torch.no_grad():
        logits, targets, pred_plan, _ = model(batch)

    n_front = int(model.wp_token.shape[0]) + 2 + 1  # wp/path tokens + speed_limit + route + sign
    seq_len = n_front + n_objects + 1
    rows = logits[0][:, 0].tolist()  # MeanHead passes the position through

    print(f"PlanT dir            : {plant}")
    print(f"trunk                : PositionStub"
          f"{' (transformers stubbed too)' if stubbed else ''}; "
          f"BEV token disabled, which only shifts every position by 1")
    print(f"sequence length      : {seq_len} = {n_front} front + {n_objects} objects + 1 speed_token")
    print(f"object positions     : {list(range(n_front, n_front + n_objects))}")
    print(f"speed_token position : {seq_len - 1}")
    print(f"logits taken from    : {[int(round(v)) for v in rows]}")
    print(f"targets refer to     : objects 0..{n_objects - 1} "
          f"(positions {list(range(n_front, n_front + n_objects))})")

    expected = list(range(n_front, n_front + n_objects))
    got = [int(round(v)) for v in rows]
    if got == expected:
        print("\nALIGNED: every object token is supervised with its own future box.")
        return
    shift = got[0] - expected[0]
    print(f"\nMISALIGNED by {shift:+d}: object token j{shift:+d} carries object j's target.")
    if got[-1] == seq_len - 1:
        head_pos = "the trailing speed_token"
        print(f"The last supervised position is {head_pos} -- the same token "
              f"ego_speed_classifier reads (model.py: pred_speed = x[:, -1]).")
    sys.exit(1)


if __name__ == "__main__":
    main()
