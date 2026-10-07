#!/usr/bin/env python3
"""make_figures.py — Vẽ biểu đồ và tạo ảnh demo/failure từ kết quả sweep.

  python src/make_figures.py

Đọc các CSV trong results/ và ghi ảnh vào results/figures/.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starter.datasets import load_frame  # noqa: E402
from starter.projection import (  # noqa: E402
    cam_to_image,
    draw_box2d,
    overlay_points,
    perturb_extrinsic,
    velo_to_cam,
)

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def col(rows, key, value_col, value, dataset=None):
    xs, ys = [], []
    for r in rows:
        if abs(float(r["value"]) - value) > 1e-9:
            continue
        if dataset and r.get("dataset") != dataset:
            continue
        xs.append(float(r["value"]))
        ys.append(float(r[value_col]))
    return np.array(xs), np.array(ys)


def agg_by_value(rows, value_col):
    """Gộp nhiều frame (mỗi frame 1 dòng) -> trung bình theo value."""
    vals = sorted({float(r["value"]) for r in rows})
    out = []
    for v in vals:
        ys = [float(r[value_col]) for r in rows if abs(float(r["value"]) - v) < 1e-9]
        out.append((v, float(np.nanmean(ys))))
    out = np.array(out)
    return out[:, 0], out[:, 1]


def fig_yaw_curve() -> None:
    rows = read_csv(ROOT / "results" / "yaw_perturb_sweep.csv")
    x, box = agg_by_value(rows, "box_retention")
    _, edge = agg_by_value(rows, "edge_retention")
    _, near = agg_by_value(rows, "box_retention_near")
    _, far = agg_by_value(rows, "box_retention_far")

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, box * 100, "-o", label="Box-retention (tất cả)")
    plt.plot(x, far * 100, "--s", label="Box-retention, xa 20–40 m")
    plt.plot(x, near * 100, "--^", label="Box-retention, gần <10 m")
    plt.plot(x, edge * 100, "-d", label="Edge-retention (Canny)")
    plt.axhline(90, color="red", ls=":", lw=1, label="Ngưỡng phát hiện drift (90%)")
    plt.axvline(1.0, color="gray", ls=":", lw=1)
    plt.annotate("yaw = 1°", (1.0, 20), rotation=90, color="gray", va="bottom")
    plt.xlabel("Lệch yaw (độ)")
    plt.ylabel("Tỉ lệ điểm còn khớp (%)")
    plt.title("Topic A — Alignment score theo mức lệch yaw (KITTI, 6 frame)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "yaw_retention_curve.png", dpi=130)
    plt.close()


def fig_axis_compare() -> None:
    axes = [("yaw", "yaw_perturb_sweep.csv", "độ"),
            ("pitch", "pitch_perturb_sweep.csv", "độ"),
            ("roll", "roll_perturb_sweep.csv", "độ"),
            ("tx", "tx_perturb_sweep.csv", "m")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, (name, fn, unit) in zip(axs, axes):
        rows = read_csv(ROOT / "results" / fn)
        x, box = agg_by_value(rows, "box_retention")
        _, edge = agg_by_value(rows, "edge_retention")
        ax.plot(x, box * 100, "-o", label="box-ret")
        ax.plot(x, edge * 100, "-d", label="edge-ret")
        ax.axhline(90, color="red", ls=":", lw=1)
        ax.set_title(f"{name}")
        ax.set_xlabel(f"mức lệch ({unit})")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Tỉ lệ còn khớp (%)")
    axs[0].legend(fontsize=8)
    fig.suptitle("Box-retention và edge-retention theo 4 trục perturb (KITTI)")
    fig.tight_layout()
    fig.savefig(FIG / "perturb_axis_compare.png", dpi=130)
    plt.close(fig)


def fig_dataset_compare() -> None:
    rows = read_csv(ROOT / "results" / "dataset_compare.csv")
    plt.figure(figsize=(7, 4.5))
    for ds, mk in (("kitti", "o"), ("nuscenes", "s")):
        x, box = agg_by_value([r for r in rows if r["dataset"] == ds], "box_retention")
        _, edge = agg_by_value([r for r in rows if r["dataset"] == ds], "edge_retention")
        plt.plot(x, box * 100, f"-{mk}", label=f"{ds} — box-ret")
        plt.plot(x, edge * 100, f"--{mk}", label=f"{ds} — edge-ret")
    plt.axhline(90, color="red", ls=":", lw=1, label="ngưỡng 90%")
    plt.xlabel("Lệch yaw (độ)")
    plt.ylabel("Tỉ lệ còn khớp (%)")
    plt.title("Cùng thí nghiệm trên KITTI (64 beam) và nuScenes (32 beam)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "dataset_compare.png", dpi=130)
    plt.close()


def _project(frame, **perturb):
    calib = perturb_extrinsic(frame["calib"], **perturb)
    uv, depth, mask = cam_to_image(velo_to_cam(frame["points"][:, :3], calib), calib.P2, frame["image"].shape)
    return uv, depth, mask


def _overlay(frame, uv, depth):
    vis = overlay_points(frame["image"], uv, depth)
    for obj in frame["labels"]:
        vis = draw_box2d(vis, obj.bbox, label=obj.type)
    return vis


def fig_demo_grid() -> None:
    frame_id = "000011"
    fr = load_frame(str(ROOT / "data" / "kitti_mini"), frame_id)
    panels = []
    for deg in (0.0, 1.0, 3.0):
        uv, depth, _ = _project(fr, yaw_deg=deg)
        vis = _overlay(fr, uv, depth)
        cv2.putText(vis, f"yaw = {deg:g} deg", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(vis, f"yaw = {deg:g} deg", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        panels.append(vis)
        cv2.imwrite(str(FIG / f"demo_projection_yaw{int(deg)}deg_{frame_id}.png"), vis)
    grid = np.vstack(panels)
    cv2.imwrite(str(FIG / f"demo_projection_grid_{frame_id}.png"), grid)


def fig_failure_tx() -> None:
    """Failure: dịch ngang 10 cm không bị box-retention phát hiện."""
    frame_id = "000004"
    fr = load_frame(str(ROOT / "data" / "kitti_mini"), frame_id)
    base = _overlay(fr, *_project(fr)[:2])
    uv_tx, depth_tx, mask_tx = _project(fr, t_xyz_m=(0.10, 0.0, 0.0))
    shifted = _overlay(fr, uv_tx, depth_tx)
    cv2.putText(base, "baseline (drift = 0)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(shifted, "tx = 10 cm (drift that)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    grid = np.vstack([base, shifted])

    # zoom vào object xa nhất để thấy điểm gần như không dịch trong box.
    far_objs = sorted(fr["labels"], key=lambda o: -float(o.location[2]))
    obj = far_objs[0]
    x1, y1, x2, y2 = [int(v) for v in obj.bbox]
    pad = 25
    xa, ya = max(0, x1 - pad), max(0, y1 - pad)
    xb, yb = min(base.shape[1], x2 + pad), min(base.shape[0], y2 + pad)
    crop_b = cv2.resize(base[ya:yb, xa:xb], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    crop_s = cv2.resize(shifted[ya:yb, xa:xb], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    zoom = np.hstack([crop_b, crop_s])
    cv2.imwrite(str(FIG / "fail_01_tx10cm_not_detected.png"), grid)
    cv2.imwrite(str(FIG / "fail_01_tx10cm_zoom.png"), zoom)


def main() -> None:
    fig_yaw_curve()
    fig_axis_compare()
    fig_dataset_compare()
    fig_demo_grid()
    fig_failure_tx()
    print("Đã ghi các hình vào", FIG)


if __name__ == "__main__":
    main()
