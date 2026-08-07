#!/usr/bin/env python3
"""Print the batch PlanT2 actually receives, built from the cached training data.

Goes through PlanTDataset + generate_batch — the same path the trainer uses — so
what it prints is the model input, not the raw boxes on disk. Answers "did the
ego see other vehicles / the sign during finetuning" without guessing.

  python3 dump_model_input.py --split <split_root> --plant-dir <repo>/plant2/PlanT
  python3 dump_model_input.py --split ... --index 5 --samples 3
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path


CLASS_NAMES = {0: "padding", 1: "car", 2: "walker", 3: "static",
               4: "stop_sign", 5: "traffic_light", 6: "emergency"}


def build_cfg(plant_dir: Path, split_root: Path):
    """Minimal cfg with the fields PlanTDataset touches (model/PlanT.yaml + train)."""
    from omegaconf import OmegaConf

    model_yaml = plant_dir / "config" / "model" / "PlanT.yaml"
    cfg_model = OmegaConf.load(model_yaml)
    cfg = OmegaConf.create({"model": cfg_model})
    # Fields the dataset reads that live outside model/PlanT.yaml.
    cfg.model.training.setdefault("filter_routes", False)
    cfg.model.training.setdefault("augment", False)
    cfg.model.training.setdefault("augment_parked", False)
    cfg.model.training.setdefault("input_bev", False)  # skip PNG decode for speed
    cfg.model.training.setdefault("seq_len", 1)
    return cfg


def describe(batch, sign_codes) -> None:
    import torch

    idxs = batch["idxs"][0]
    pool = batch["x_objs"]
    rows = [pool[i] for i in idxs.tolist() if i > 0]

    print(f"  объектов в кадре: {len(rows)}")
    counts = collections.Counter()
    for r in rows:
        cid = int(r[0].item())
        name = CLASS_NAMES.get(cid)
        if name is None:
            k = cid - 7
            name = f"знак {sign_codes[k]}" if 0 <= k < len(sign_codes) else f"class{cid}"
        counts[name] += 1
        print(f"    class={cid:<3} {name:<16} x={r[1].item():+7.1f} y={r[2].item():+7.1f} "
              f"yaw={r[3].item():+6.0f} spd/id={r[4].item():+6.1f} "
              f"ext=({r[5].item():.1f},{r[6].item():.1f})")
    print(f"  состав: {dict(counts) or 'ПУСТО'}")

    for key in ("target_speed", "speed_limit", "sign_id", "ego_speed"):
        if key in batch:
            v = batch[key][0]
            print(f"  {key}: {v.item() if v.ndim == 0 else v.tolist()}")
    if "route" in batch:
        rt = batch["route"][0]
        print(f"  route: точек {len(rt)}, первая {rt[0].tolist()}, "
              f"последняя {rt[-1].tolist()}")
    if "waypoints" in batch:
        wp = batch["waypoints"][0]
        print(f"  waypoints: {len(wp)} шт, последняя {wp[-1].tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True,
                    help="корень сплита (…/train) или каталог data внутри него")
    ap.add_argument("--plant-dir", required=True, help="…/plant2/PlanT")
    ap.add_argument("--index", type=int, default=0, help="номер первого сэмпла")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1, help="шаг между сэмплами")
    args = ap.parse_args()

    plant_dir = Path(args.plant_dir).resolve()
    sys.path.insert(0, str(plant_dir))
    sys.path.insert(0, str(plant_dir / "util"))

    root = Path(args.split).resolve()
    if (root / "data").is_dir():
        root = root / "data"

    from dataset import PlanTDataset, generate_batch  # noqa: E402
    from util.sign_id import SIGN_CODES  # noqa: E402

    cfg = build_cfg(plant_dir, root)
    ds = PlanTDataset(str(root), cfg)
    print(f"\nсэмплов в выборке: {len(ds)}\n")

    for k in range(args.samples):
        i = args.index + k * args.stride
        if i >= len(ds):
            break
        sample = ds[i]
        batch = generate_batch([sample])
        print(f"=== сэмпл {i}")
        describe(batch, SIGN_CODES)
        print()


if __name__ == "__main__":
    main()
