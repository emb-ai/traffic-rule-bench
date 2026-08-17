#!/usr/bin/env python3
"""Put the training object sequence and the eval object sequence side by side.

Both are built from the SAME dumped boxes file, so any difference is the
encoder's, not the scene's. What the model sees depends on token order and
sequence length, because the trunk is a BERT and its absolute position
embeddings are added to `inputs_embeds` (model.py: word_embeddings removed,
position_embeddings kept).

  python3 compare_train_eval_objects.py --route <split>/train/data/<route>
  python3 compare_train_eval_objects.py --route <route> --frame 120

The training column mirrors PlanTDataset.__getitem__ (dataset.py:409-493); the
eval column calls the real `boxes_to_objects_list`. For an authoritative
training-side dump on the cluster use dump_model_input.py, which runs the
dataset itself.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path

TRB_ROOT = Path(__file__).resolve().parents[3]  # tools/ -> pipeline -> scripts -> repo
PLANT_T = TRB_ROOT / "plant2" / "PlanT"
ENCODER = TRB_ROOT / "metadrive" / "metadrive" / "policy" / "metadrive_obs_to_plant2.py"

MAX_DISTANCE = 50.0
RANGE_FACTOR_FRONT = 2.0
SIGN_RADIUS = 30.0


def load_encoder():
    """Import the eval encoder by path: `import metadrive` would pull in panda3d."""
    spec = importlib.util.spec_from_file_location("_plant2_encoder", ENCODER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_boxes(route: Path, frame: int | None) -> tuple[list, Path]:
    files = sorted((route / "boxes").glob("*.json.gz"))
    if not files:
        raise SystemExit(f"no boxes/*.json.gz under {route}")
    path = files[frame] if frame is not None else files[len(files) // 2]
    with gzip.open(path, "rt") as fh:
        return json.load(fh), path


def train_order(boxes, type_nums, sign_like, car_types) -> list:
    """Mirror of dataset.py: cars in dump order, then static-ish in dump order."""
    labels = boxes[1:]  # ego is first

    def too_far(x) -> bool:
        if "position" not in x:
            return False
        px, py, pz = x["position"]
        if x["class"] in {"traffic_light"} | set(sign_like):
            return px * px + py * py > SIGN_RADIUS ** 2 or abs(pz) > 30
        div = RANGE_FACTOR_FRONT ** 2 if px > 0 else 1.0
        return px * px / div + py * py > MAX_DISTANCE ** 2 or abs(pz) > 30

    def cls_key(x) -> str:
        return x["class"] if x["class"] in type_nums else str(x["class"]).lower()

    kept = [x for x in labels if not too_far(x) and cls_key(x) in type_nums]
    cars = [x for x in kept if cls_key(x) in car_types]
    staticish = []
    for x in kept:
        key = cls_key(x)
        if key in car_types:
            continue
        if key == "traffic_light" and not (
                x.get("state") in ("Red", "Yellow") and x.get("affects_ego")):
            continue
        if (key in sign_like or str(x["class"]).lower() in sign_like) and not x.get("affects_ego"):
            continue
        staticish.append(x)
    return [(cls_key(x), float(x["position"][0]), float(x["position"][1])) for x in cars + staticish]


def sign_index(rows, sign_like) -> str:
    hits = [i for i, r in enumerate(rows) if r[0] in sign_like]
    return ", ".join(str(i) for i in hits) if hits else "нет"


def show(title: str, rows, sign_like, n_front: int, seq_objects: int) -> None:
    print(f"\n{title}")
    for i, (cls, x, y) in enumerate(rows):
        mark = "  <== знак" if cls in sign_like else ""
        print(f"  [{i:2d}] pos={n_front + i:<3d} {cls:<12} x={x:+7.1f} y={y:+7.1f}{mark}")
    print(f"  объектов: {len(rows)}   индекс знака: {sign_index(rows, sign_like)}")
    print(f"  длина последовательности: {n_front} + {seq_objects} + 1 = "
          f"{n_front + seq_objects + 1}, speed_token на позиции {n_front + seq_objects}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", required=True, help="route dir containing boxes/")
    ap.add_argument("--frame", type=int, default=None, help="index into boxes/ (default: middle)")
    ap.add_argument("--max-objects", type=int, default=30, help="eval cap (adapter passes 30)")
    ap.add_argument("--n-front", type=int, default=32,
                    help="tokens before the objects: 28 wp/path + BEV + sign + speed_limit + route")
    args = ap.parse_args()

    sys.path.insert(0, str(PLANT_T))
    enc = load_encoder()
    type_nums, sign_like, car_types = enc._get_type_nums_and_sign_like()

    boxes, path = load_boxes(Path(args.route), args.frame)
    print(f"кадр: {path}")
    print(f"боксов в кадре: {len(boxes)} (первый — эго)")

    # "car" and "static_car" share class id 1.0 — keep the first name, not the last.
    id2name: dict[float, str] = {}
    for name, num in type_nums.items():
        id2name.setdefault(float(num), name)
    tr = train_order(boxes, type_nums, sign_like, car_types)
    ev = [(id2name.get(float(o[0]), f"class{o[0]:.0f}"), o[1], o[2])
          for o in enc.boxes_to_objects_list(boxes, max_objects=args.max_objects)]

    show("ОБУЧЕНИЕ (dataset.py: машины, затем статика и знаки; без сортировки)",
         tr, sign_like, args.n_front, len(tr))
    import os

    order = (os.environ.get("PLANT2_OBJ_ORDER") or "dist").strip().lower()
    seq_fit = (os.environ.get("PLANT2_SEQ_FIT") or "").strip() in ("1", "true", "True")
    show(f"ЭВАЛ (metadrive_obs_to_plant2.py: PLANT2_OBJ_ORDER={order}, "
         f"PLANT2_SEQ_FIT={'1' if seq_fit else '0'})",
         ev, sign_like, args.n_front, len(ev) if seq_fit else args.max_objects)

    print("\nитог:")
    print(f"  порядок знака:      обучение {sign_index(tr, sign_like)}  "
          f"vs эвал {sign_index(ev, sign_like)}")
    print(f"  позиция speed_token: обучение {args.n_front + len(tr)}  "
          f"vs эвал {args.n_front + (len(ev) if seq_fit else args.max_objects)}")
    print("  (в обучении maxseq — максимум по батчу, не по сэмплу: "
          "число выше — нижняя граница)")


if __name__ == "__main__":
    main()
