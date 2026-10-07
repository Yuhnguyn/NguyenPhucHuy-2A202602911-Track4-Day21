import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from starter.datasets import load_frame
from starter.projection import project_velo_to_image, perturb_extrinsic
from src.projection_qa import _valid_positions, _points_in_boxes, _retention


def per_object(data_root, fid, **pk):
    fr = load_frame(data_root, fid)
    buv, _, bm = project_velo_to_image(fr["points"], perturb_extrinsic(fr["calib"]), fr["image"].shape)
    boxes = _points_in_boxes(buv, bm, fr["labels"])
    calib = perturb_extrinsic(fr["calib"], **pk)
    uv, _, mask = project_velo_to_image(fr["points"], calib, fr["image"].shape)
    pos = _valid_positions(mask)
    out = []
    for gi, o in zip(boxes, fr["labels"]):
        r = _retention(gi, uv, pos, o.bbox)
        out.append((o.type, round(float(o.location[2]), 1), len(gi),
                    round(r, 3) if not np.isnan(r) else None))
    return out


if __name__ == "__main__":
    for fid in ["000019", "000004", "000031", "000061"]:
        for pk in [dict(yaw_deg=1.0), dict(yaw_deg=3.0), dict(t_xyz_m=(0.10, 0, 0))]:
            print(fid, pk, per_object("data/kitti_mini", fid, **pk))
        print()
