#!/usr/bin/env python3
"""Does the model read the sign token, or just the scene?

Runs a checkpoint twice over the same frames, changing nothing but the sign
token: 3.24-at-20 against 3.24-at-60. Geometry, objects, route and speed limit
are identical, so any change in the predicted speed can only come from the
token. No change means the sign is decoration, whatever the losses say.

This reads a checkpoint offline. It adds no rule on top of the model and does
not touch the benchmark's own evaluation.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, help="split root; its val/data is used")
    ap.add_argument("--plan-t", default=os.environ.get("PLAN_T", ""),
                    help="path to plant2/PlanT (defaults to $PLAN_T)")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--from-value", type=float, default=20.0)
    ap.add_argument("--to-value", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    plan_t = Path(args.plan_t or (Path(__file__).resolve().parents[3] / "plant2" / "PlanT"))
    sys.path.insert(0, str(plan_t))

    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from dataset import PlanTDataset, generate_batch
    from lit_module import LitHFLM
    from plant_variables import PlanTVariables
    from util.sign_id import sign_code_to_id

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["hyper_parameters"]["cfg"]) if "hyper_parameters" in ckpt else None
    if cfg is None:
        print("checkpoint carries no cfg; pass one through hydra instead", file=sys.stderr)
        return 2

    ds = PlanTDataset(str(Path(args.split) / "val" / "data"), cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=generate_batch, num_workers=0)

    model = LitHFLM.load_from_checkpoint(args.ckpt, cfg=cfg, strict=False, map_location="cpu")
    model.eval().to(args.device)

    speeds = torch.tensor(PlanTVariables.target_speeds, device=args.device)
    id_from = sign_code_to_id("3.24", args.from_value)
    id_to = sign_code_to_id("3.24", args.to_value)
    print(f"sign_id {id_from} (3.24@{args.from_value:g}) -> {id_to} (3.24@{args.to_value:g})")

    per_plate = defaultdict(lambda: [0, 0.0, 0.0])
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            sel = batch["sign_id"].reshape(-1) == id_from
            if not bool(sel.any()):
                continue

            def predict(sign_ids):
                b = dict(batch)
                b["sign_id"] = sign_ids
                _, _, plan, _ = model(b)
                return speeds[plan[2].argmax(dim=-1)] * 3.6

            base = predict(batch["sign_id"])
            swapped_ids = batch["sign_id"].clone()
            swapped_ids[sel] = id_to
            swapped = predict(swapped_ids)

            plate = args.from_value
            acc = per_plate[plate]
            acc[0] += int(sel.sum())
            acc[1] += float(base[sel].sum())
            acc[2] += float(swapped[sel].sum())

    if not per_plate:
        print(f"no frames carried sign_id {id_from}; nothing to compare")
        return 1

    print(f"\n{'plate':>6} {'frames':>7} {'pred@from':>10} {'pred@to':>9} {'delta':>8}")
    for plate, (n, s_base, s_swap) in sorted(per_plate.items()):
        print(f"{plate:>6.0f} {n:>7d} {s_base / n:>10.2f} {s_swap / n:>9.2f} "
              f"{(s_swap - s_base) / n:>+8.2f}")
    print("\nA delta of zero means the token is not used. A positive delta means "
          "the model drives faster when the plate says a higher number, which is "
          "the binding this whole exercise is after.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
