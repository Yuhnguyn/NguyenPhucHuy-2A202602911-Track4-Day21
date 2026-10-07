#!/usr/bin/env python3
"""projection_qa.py — Tool tái sử dụng để kiểm định calibration LiDAR-camera (Topic A).

Chức năng:
  sweep             Quét một tham số perturb (yaw/pitch/roll/tx/ty/tz) qua nhiều mức,
                    đo 2 metric alignment: điểm-trong-box và điểm-trên-cạnh (Canny).
  compare-datasets  Chạy cùng một sweep trên 2 dataset (KITTI vs nuScenes) để so sánh.
  latency           Benchmark độ trễ projection (p50/p95, bỏ lần chạy đầu).
  audit             Phát hiện lỗi cài sẵn trong một thư mục dữ liệu (NaN, timestamp, sector...).

Ví dụ:
  python src/projection_qa.py sweep --data-root data/kitti_mini \
      --frames 000011 000004 000019 000031 --axis yaw --values 0 0.5 1 2 3
  python src/projection_qa.py compare-datasets --axis yaw --values 0 1 2 3
  python src/projection_qa.py latency --data-root data/kitti_mini --frame 000011 -n 40
  python src/projection_qa.py audit --data-root data/synthetic

Ghi chú: metric "điểm-trong-box" lấy tại baseline (drift = 0) các điểm đang nằm trong
2D box của từng object, rồi đo tỉ lệ các điểm đó *còn* nằm trong đúng box đó sau khi
làm lệch calibration. Metric "điểm-trên-cạnh" làm tương tự với bản đồ cạnh Canny.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Cho phép chạy trực tiếp: python src/projection_qa.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starter.datasets import list_frames, load_frame, load_points  # noqa: E402
from starter.projection import (  # noqa: E402
    draw_box2d,
    overlay_points,
    perturb_extrinsic,
    project_velo_to_image,
)

PERTURB_FNS = {
    "yaw": lambda v: {"yaw_deg": v},
    "pitch": lambda v: {"pitch_deg": v},
    "roll": lambda v: {"roll_deg": v},
    "tx": lambda v: {"t_xyz_m": (v, 0.0, 0.0)},
    "ty": lambda v: {"t_xyz_m": (0.0, v, 0.0)},
    "tz": lambda v: {"t_xyz_m": (0.0, 0.0, v)},
}


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _valid_positions(mask: np.ndarray) -> np.ndarray:
    """Bảng tra: chỉ số toàn cục của điểm -> vị trí trong mảng uv (valid). -1 nếu không hợp lệ."""
    pos = np.full(mask.shape[0], -1, dtype=np.int64)
    valid = np.where(mask)[0]
    pos[valid] = np.arange(valid.size)
    return pos


def _points_in_boxes(uv: np.ndarray, mask: np.ndarray, labels) -> list[np.ndarray]:
    """Với mỗi object, trả về chỉ số toàn cục các điểm nằm trong 2D box của nó."""
    valid_idx = np.where(mask)[0]
    out = []
    if uv.size == 0:
        return [np.zeros(0, dtype=np.int64) for _ in labels]
    for obj in labels:
        x1, y1, x2, y2 = obj.bbox
        sel = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        out.append(valid_idx[sel])
    return out


def _retention(global_idx: np.ndarray, uv: np.ndarray, pos: np.ndarray, box) -> float:
    """Tỉ lệ các điểm (chỉ số toàn cục) còn chiếu vào trong `box` ở lần chiếu mới."""
    if global_idx.size == 0:
        return float("nan")
    p = pos[global_idx]
    ok = p >= 0
    if not ok.any():
        return 0.0
    u = uv[p[ok], 0]
    v = uv[p[ok], 1]
    x1, y1, x2, y2 = box
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    kept = inside.sum()
    return float(kept) / float(global_idx.size)


def _edge_map(image: np.ndarray, radius: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    k = 2 * radius + 1
    return cv2.dilate(edges, np.ones((k, k), np.uint8))


def _edge_points(uv: np.ndarray, mask: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Chỉ số toàn cục các điểm chiếu trúng cạnh ảnh."""
    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    u = np.clip(uv[:, 0].astype(int), 0, edge.shape[1] - 1)
    v = np.clip(uv[:, 1].astype(int), 0, edge.shape[0] - 1)
    on = edge[v, u] > 0
    return valid_idx[on]


def _bucket(z_cam: float) -> str:
    if z_cam < 10:
        return "near"
    if z_cam < 20:
        return "mid"
    if z_cam < 40:
        return "far"
    return "veryfar"


def frame_metrics(frame: dict, perturb_kwargs: dict, edge_radius: int = 3) -> dict:
    """Tính metric cho một frame với một mức perturb cụ thể."""
    image = frame["image"]
    baseline = perturb_extrinsic(frame["calib"])  # drift = 0
    base_uv, _, base_mask = project_velo_to_image(frame["points"], baseline, image.shape)
    pos0 = _valid_positions(base_mask)
    boxes = _points_in_boxes(base_uv, base_mask, frame["labels"])

    edge = _edge_map(image, edge_radius)
    edge_idx0 = _edge_points(base_uv, base_mask, edge)

    calib = perturb_extrinsic(frame["calib"], **perturb_kwargs)
    uv, _, mask = project_velo_to_image(frame["points"], calib, image.shape)
    pos = _valid_positions(mask)

    n_points = int(frame["points"].shape[0])
    pct_in_fov = float(mask.mean())

    # Điểm còn nằm trong box nào đó (không phân biệt object).
    in_any = np.zeros(0, dtype=bool)
    for gi, obj in zip(boxes, frame["labels"]):
        p = pos[gi]
        ok = p >= 0
        if not ok.any():
            continue
        u = uv[p[ok], 0]
        v = uv[p[ok], 1]
        x1, y1, x2, y2 = obj.bbox
        ins = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        in_any = np.concatenate([in_any, ins])
    pct_in_box = float(in_any.mean()) if in_any.size else float("nan")

    # Box retention theo từng object, gộp theo khoảng cách.
    per_bucket: dict[str, list[float]] = {"near": [], "mid": [], "far": [], "veryfar": []}
    all_ret = []
    for gi, obj in zip(boxes, frame["labels"]):
        r = _retention(gi, uv, pos, obj.bbox)
        if not np.isnan(r):
            all_ret.append(r)
            per_bucket[_bucket(float(obj.location[2]))].append(r)
    box_ret_all = float(np.mean(all_ret)) if all_ret else float("nan")

    p = pos[edge_idx0]
    ok = p >= 0
    if ok.any():
        u = np.clip(uv[p[ok], 0].astype(int), 0, edge.shape[1] - 1)
        v = np.clip(uv[p[ok], 1].astype(int), 0, edge.shape[0] - 1)
        edge_ret = float((edge[v, u] > 0).sum()) / float(edge_idx0.size)
    else:
        edge_ret = 0.0

    def bmean(name):
        vals = per_bucket[name]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_points": n_points,
        "pct_in_fov": pct_in_fov,
        "pct_in_box": pct_in_box,
        "box_retention": box_ret_all,
        "box_retention_near": bmean("near"),
        "box_retention_far": bmean("far"),
        "edge_retention": edge_ret,
        "n_objects": len(frame["labels"]),
    }


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #
def _load_frames(data_root: str, frames: list[str], camera: str, ego: bool) -> dict[str, dict]:
    ids = frames or list_frames(data_root)
    out = {}
    for fid in ids:
        kw = {}
        if camera:
            kw["camera"] = camera
        if not ego:
            kw["use_ego_motion"] = False
        try:
            out[fid] = load_frame(data_root, fid, **kw)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {fid}: {exc}")
    return out


def cmd_sweep(args) -> None:
    frames = _load_frames(args.data_root, args.frames, args.camera, args.ego)
    if not frames:
        raise SystemExit("Không có frame nào được load.")
    fn = PERTURB_FNS[args.axis]
    rows = []
    for value in args.values:
        kwargs = fn(float(value))
        recs = [frame_metrics(fr, kwargs, args.edge_radius) for fr in frames.values()]
        for fid, m in zip(frames.keys(), recs):
            rows.append({"axis": args.axis, "value": value, "frame_id": fid, **m})
        agg = {k: float(np.nanmean([r[k] for r in recs])) for k in
               ("pct_in_fov", "pct_in_box", "box_retention", "box_retention_near",
                "box_retention_far", "edge_retention")}
        print(f"{args.axis}={value:<5} | in_fov={agg['pct_in_fov']:.3f} "
              f"in_box={agg['pct_in_box']:.3f} box_ret={agg['box_retention']:.3f} "
              f"(near={agg['box_retention_near']:.3f} far={agg['box_retention_far']:.3f}) "
              f"edge_ret={agg['edge_retention']:.3f}")
    _write_csv(args.csv, rows)
    print(f"-> {args.csv}")


def cmd_compare_datasets(args) -> None:
    rows = []
    for root, tag, frames in (("data/kitti_mini", "kitti", args.kitti_frames),
                              ("data/nuscenes_mini_subset", "nuscenes", None)):
        print(f"== {tag} ==")
        args.data_root = root
        args.frames = frames or []
        frames_loaded = _load_frames(root, args.frames, args.camera, args.ego)
        fn = PERTURB_FNS[args.axis]
        for value in args.values:
            recs = [frame_metrics(fr, fn(float(value)), args.edge_radius) for fr in frames_loaded.values()]
            agg = {k: float(np.nanmean([r[k] for r in recs])) for k in
                   ("pct_in_fov", "pct_in_box", "box_retention", "box_retention_far", "edge_retention")}
            rows.append({"dataset": tag, "n_frames": len(recs), "axis": args.axis, "value": value, **agg})
            print(f"  {args.axis}={value:<5} box_ret={agg['box_retention']:.3f} edge_ret={agg['edge_retention']:.3f}")
    _write_csv(args.csv, rows)


def cmd_latency(args) -> None:
    fr = load_frame(args.data_root, args.frame)
    calib = fr["calib"]
    shape = fr["image"].shape
    times = []
    for i in range(args.n):
        t0 = time.perf_counter()
        project_velo_to_image(fr["points"], calib, shape)
        dt = (time.perf_counter() - t0) * 1000.0
        if i > 0:  # bỏ lần chạy đầu (warm-up)
            times.append(dt)
    times = np.array(times)
    p50, p95 = float(np.percentile(times, 50)), float(np.percentile(times, 95))
    print(f"n_points={fr['points'].shape[0]} runs={len(times)} p50={p50:.2f} ms p95={p95:.2f} ms")
    _write_csv(args.csv, [{"data_root": args.data_root, "frame_id": args.frame,
                           "n_points": fr["points"].shape[0], "n_runs": len(times),
                           "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
                           "mean_ms": round(float(times.mean()), 3),
                           "min_ms": round(float(times.min()), 3),
                           "max_ms": round(float(times.max()), 3)}])


def cmd_audit(args) -> None:
    """Phát hiện lỗi dữ liệu: NaN/Inf, timestamp bất thường, sector/beam dropout.

    Cách phát hiện sector dropout: dựng histogram azimuth 36 bin cho từng frame, lấy
    histogram trung vị qua các frame làm "khuôn" tham chiếu, rồi tìm các bin bị hụt
    mạnh (< 50% tham chiếu). Một frame có cụm bin liền nhau hụt mạnh = sector bị mất.
    """
    frames = list_frames(args.data_root)
    ts_path = Path(args.data_root) / "training" / "timestamps.txt"
    deltas = None
    if ts_path.exists():
        ts = np.array([float(x) for x in ts_path.read_text().split()])
        deltas = np.diff(ts)

    n_bins = 36
    hists, raw_pts = {}, {}
    for fid in frames:
        raw = load_points(args.data_root, fid)
        raw_pts[fid] = raw
        finite = np.isfinite(raw[:, :3]).all(axis=1)
        pts = raw[finite]
        az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        hists[fid], _ = np.histogram(az, bins=n_bins, range=(-180, 180))
    ref = np.median(np.vstack(list(hists.values())), axis=0)
    base_n = float(np.median([len(v) for v in raw_pts.values()]))

    rows = []
    for fid in frames:
        raw = raw_pts[fid]
        finite = np.isfinite(raw[:, :3]).all(axis=1)
        h = hists[fid]
        deficit = h < 0.5 * np.maximum(ref, 1)
        n_deficit = int(deficit.sum())
        n_nan = int((~finite).sum())
        note = []
        if n_nan:
            note.append(f"NaN points ({n_nan})")
        if len(raw) < 0.95 * base_n:
            note.append(f"dropout (-{100*(1-len(raw)/base_n):.0f}% vs median)")
        if n_deficit >= 2:
            note.append(f"sector dropout ({n_deficit} bins)")
        rows.append({"frame_id": fid, "n_points": len(raw), "n_nan": n_nan,
                     "n_deficit_bins": n_deficit,
                     "ref_median_pts_per_bin": round(float(np.median(ref)), 1)})
        print(f"{fid}  n={len(raw):>7}  nan={n_nan:>3}  deficit_bins={n_deficit:>2}  "
              f"| {', '.join(note) if note else 'OK'}")

    if deltas is not None:
        print("timestamp deltas (s):", deltas.round(3).tolist())
        med = float(np.median(deltas))
        bad = [(i, round(float(d), 3)) for i, d in enumerate(deltas) if d > 1.5 * med]
        print("timestamp gaps > 1.5x median:", bad if bad else "none")
    _write_csv(args.csv, rows)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--axis", default="yaw", choices=sorted(PERTURB_FNS))
        p.add_argument("--values", type=float, nargs="+", default=[0, 0.5, 1, 2, 3])
        p.add_argument("--edge-radius", type=int, default=3)
        p.add_argument("--camera", default="CAM_FRONT", help="camera nuScenes (mặc định CAM_FRONT)")
        p.add_argument("--no-ego-motion", dest="ego", action="store_false", help="tắt bù chuyển động nuScenes")
        p.set_defaults(ego=True)

    s = sub.add_parser("sweep", help="quét một tham số perturb")
    s.add_argument("--data-root", default="data/kitti_mini")
    s.add_argument("--frames", nargs="*", default=[])
    s.add_argument("--csv", default="results/yaw_perturb_sweep.csv")
    common(s)
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser("compare-datasets", help="so sánh KITTI vs nuScenes")
    c.add_argument("--kitti-frames", nargs="*", default=["000011", "000004", "000031"])
    c.add_argument("--csv", default="results/dataset_compare.csv")
    common(c)
    c.set_defaults(func=cmd_compare_datasets)

    l = sub.add_parser("latency", help="benchmark độ trễ projection")
    l.add_argument("--data-root", default="data/kitti_mini")
    l.add_argument("--frame", default="000011")
    l.add_argument("-n", type=int, default=40)
    l.add_argument("--csv", default="results/latency.csv")
    l.set_defaults(func=cmd_latency)

    a = sub.add_parser("audit", help="phát hiện lỗi dữ liệu")
    a.add_argument("--data-root", default="data/synthetic")
    a.add_argument("--csv", default="results/synthetic_audit.csv")
    a.set_defaults(func=cmd_audit)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
