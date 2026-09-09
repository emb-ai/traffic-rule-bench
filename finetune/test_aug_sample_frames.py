"""aug_sample must reproduce what the dumper would have written from a jittered pose.

`PlanTDataset.aug_sample` re-expresses one recorded frame as if the ego had sat
`augmentation_translation` metres to its own LEFT and been rotated
`augmentation_rotation` degrees CCW -- the same virtual pose
`render_bev_plant2(lateral_offset_m=, heading_offset_rad=)` draws BEV_aug from.

The dump mixes two lateral conventions in one sample (object boxes are y=RIGHT,
route / route_original / waypoints are y=LEFT), so the only trustworthy oracle
is the dumper's own arithmetic. This test rebuilds a synthetic world, asks the
dumper formulas for the frame at the true pose and at the jittered pose, and
requires aug_sample to turn the first into the second.

Run:  python -m pytest finetune/test_aug_sample_frames.py -q
"""
import math

import numpy as np

# The exact transform in third_party/plant2/PlanT/dataset.py::aug_sample.
# Imported by source rather than by module so the test does not need torch,
# hydra or a config tree.
import ast
import pathlib

_DATASET = (pathlib.Path(__file__).resolve().parents[1]
            / "third_party" / "plant2" / "PlanT" / "dataset.py")


def _load_aug_sample():
    """Pull aug_sample out of dataset.py without importing the whole module."""
    tree = ast.parse(_DATASET.read_text(encoding="utf-8"))
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "aug_sample":
                    mod = ast.Module(body=[fn], type_ignores=[])
                    ast.fix_missing_locations(mod)
                    ns = {"np": np}
                    exec(compile(mod, str(_DATASET), "exec"), ns)
                    return ns["aug_sample"]
    raise AssertionError("aug_sample not found in dataset.py")


AUG_SAMPLE = _load_aug_sample()


class _Cfg(dict):
    """Stand-in for cfg_train: aug_sample only calls .get('input_bev', False)."""


class _Self:
    cfg_train = _Cfg()

    @staticmethod
    def quantize_box(boxes):
        return boxes           # identity: the test compares floats


# ---------------------------------------------------------------- the dumper

def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def dump_box(obj_xy, obj_heading, ego_xy, ego_heading):
    """finetune/plant2_frames.py::collect_boxes._add, positions + yaw.

    local = convert_to_local_coordinates(world - ego)  -> (forward, LEFT)
    then y is negated -> y=RIGHT; yaw is NOT mirrored.
    """
    rel = np.asarray(obj_xy, float) - np.asarray(ego_xy, float)
    c, s = math.cos(-ego_heading), math.sin(-ego_heading)
    fwd = rel[0] * c - rel[1] * s
    left = rel[0] * s + rel[1] * c
    return np.array([fwd, -left, math.degrees(wrap_to_pi(obj_heading - ego_heading))])


def dump_route_point(pt_xy, ego_xy, ego_heading):
    """finetune/plant2_frames.py::get_route -- same rotation, y kept LEFT."""
    rel = np.asarray(pt_xy, float) - np.asarray(ego_xy, float)
    c, s = math.cos(-ego_heading), math.sin(-ego_heading)
    return np.array([rel[0] * c - rel[1] * s, rel[0] * s + rel[1] * c])


def jittered_pose(ego_xy, ego_heading, translation_m, rotation_deg):
    """metadrive_obs_to_plant2.py::render_bev_plant2 -- the virtual viewpoint."""
    left = np.array([-math.sin(ego_heading), math.cos(ego_heading)])
    return (np.asarray(ego_xy, float) + left * translation_m,
            ego_heading + math.radians(rotation_deg))


# ------------------------------------------------------------------- fixture

WORLD_OBJECTS = [                      # (x, y, heading) in world coordinates
    (18.0, 3.0, 0.30),
    (25.0, -6.5, -1.10),
    (8.0, 0.5, math.pi),
    (40.0, 12.0, 2.40),
]
WORLD_ROUTE = [(5.0 + i * 1.0, 0.2 * i) for i in range(20)]
EGO_XY, EGO_HEADING = (3.0, -1.5), 0.4


def _build_sample(translation_m, rotation_deg):
    boxes = [dump_box(o[:2], o[2], EGO_XY, EGO_HEADING) for o in WORLD_OBJECTS]
    route = np.array([dump_route_point(p, EGO_XY, EGO_HEADING) for p in WORLD_ROUTE])
    return {
        # input row: [type, x, y_right, yaw_deg, speed, w, l]
        "input": [[1.0, b[0], b[1], b[2], 0.0, 2.0, 4.0] for b in boxes],
        "waypoints": route[:10].tolist(),
        "route": route.tolist(),
        "route_original": route.tolist(),
        "augmentation_translation": translation_m,
        "augmentation_rotation": rotation_deg,
    }


def _expected(translation_m, rotation_deg):
    """What the dumper WOULD have written standing at the jittered pose."""
    jxy, jh = jittered_pose(EGO_XY, EGO_HEADING, translation_m, rotation_deg)
    boxes = [dump_box(o[:2], o[2], jxy, jh) for o in WORLD_OBJECTS]
    route = np.array([dump_route_point(p, jxy, jh) for p in WORLD_ROUTE])
    return np.array(boxes), route


CASES = [(1.0, 5.0), (-1.0, -5.0), (0.75, 0.0), (0.0, 4.0), (0.0, 0.0)]


def test_augmented_sample_matches_a_real_jittered_dump():
    for t_m, r_deg in CASES:
        got = AUG_SAMPLE(_Self(), _build_sample(t_m, r_deg))
        exp_boxes, exp_route = _expected(t_m, r_deg)

        got_boxes = np.asarray(got["input"], dtype=float)
        assert np.allclose(got_boxes[:, 1:3], exp_boxes[:, 0:2], atol=1e-9), (
            f"object positions wrong for t={t_m} r={r_deg}\n"
            f"got  {got_boxes[:, 1:3]}\nwant {exp_boxes[:, 0:2]}")
        # yaw is periodic
        dy = (got_boxes[:, 3] - exp_boxes[:, 2] + 180.0) % 360.0 - 180.0
        assert np.allclose(dy, 0.0, atol=1e-9), (
            f"object yaw wrong for t={t_m} r={r_deg}: off by {dy} deg")

        for key, n in (("route", 20), ("route_original", 20), ("waypoints", 10)):
            assert np.allclose(np.asarray(got[key], dtype=float),
                               exp_route[:n], atol=1e-9), (
                f"{key} wrong for t={t_m} r={r_deg}")


def test_the_two_groups_agree_on_one_physical_point():
    """The regression the split fixes.

    Place one object exactly on a route point. After augmentation the object
    box and that route point must still describe the same physical place --
    which means the same distance from the ego, since the two are stored in
    mirrored frames. Before the fix a 1 m / 5 deg jitter drove them apart.
    """
    t_m, r_deg = 1.0, 5.0
    shared_world = WORLD_ROUTE[10]
    sample = _build_sample(t_m, r_deg)
    obj = dump_box(shared_world, 0.0, EGO_XY, EGO_HEADING)
    sample["input"].append([1.0, obj[0], obj[1], obj[2], 0.0, 2.0, 4.0])

    got = AUG_SAMPLE(_Self(), sample)
    box_xy = np.asarray(got["input"][-1][1:3], dtype=float)
    route_xy = np.asarray(got["route_original"][10], dtype=float)
    # mirror the y=right box back into the y=left frame to compare
    box_in_left_frame = np.array([box_xy[0], -box_xy[1]])
    err = float(np.linalg.norm(box_in_left_frame - route_xy))
    assert err < 1e-9, (
        f"object box and the route point on top of it ended up {err:.3f} m "
        f"apart after a {t_m} m / {r_deg} deg jitter")


if __name__ == "__main__":
    test_augmented_sample_matches_a_real_jittered_dump()
    test_the_two_groups_agree_on_one_physical_point()
    print("PASS")
