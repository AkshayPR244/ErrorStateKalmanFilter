#!/usr/bin/env python3
"""
Visualize ESKF GPS-outage stress-test results from drive_0001 and drive_0009.
Reads the _trajectory.csv files produced by `cargo run --release`.

Layout (2 sequences x 4 columns):
  [0-1] Trajectory top-down
  [2]   ATE  -- per-frame absolute position error with phase labels
  [3]   RPE  -- 10-frame relative pose error (local consistency)

CSV columns:
  frame, outage, gt_x, gt_y,
  full_gps_x, full_gps_y,
  imu_dead_x, imu_dead_y,
  aided_dead_x, aided_dead_y
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RPE_WINDOW = 10   # frames (~1 s at 10 Hz)

# ---------------------------------------------------------------------------

def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def ate(ex, ey, gx, gy):
    return np.sqrt((ex - gx)**2 + (ey - gy)**2)


def rpe_series(ex, ey, gx, gy, w=RPE_WINDOW):
    """RPE[i] = ||(est[i+w]-est[i]) - (gt[i+w]-gt[i])||, length n-w."""
    n = len(ex)
    vals = np.empty(n - w)
    for i in range(n - w):
        de = np.array([ex[i+w] - ex[i], ey[i+w] - ey[i]])
        dg = np.array([gx[i+w] - gx[i], gy[i+w] - gy[i]])
        vals[i] = np.linalg.norm(de - dg)
    return vals


def phase_rmse(err_arr, outage):
    """Return (pre, during, post) RMSE for an outage flag vector."""
    first_out = np.argmax(outage)
    last_out  = len(outage) - 1 - np.argmax(outage[::-1])
    def _r(a): return float(np.sqrt(np.mean(a**2))) if len(a) else 0.0
    return (_r(err_arr[:first_out]),
            _r(err_arr[outage == 1]),
            _r(err_arr[last_out + 1:]))


def shade(ax, frames, outage, label="GPS outage"):
    """Fill grey band where outage==1."""
    in_b, s0 = False, 0
    for i, f in enumerate(frames):
        if outage[i] and not in_b:
            s0, in_b = f, True
        elif not outage[i] and in_b:
            ax.axvspan(s0, f, color="grey", alpha=0.18, label=label)
            label, in_b = None, False
    if in_b:
        ax.axvspan(s0, frames[-1], color="grey", alpha=0.18, label=label)


# ---------------------------------------------------------------------------

SEQS = [
    ("drive_0001_trajectory.csv", "drive_0001  (108 frames, straight road)"),
    ("drive_0009_trajectory.csv", "drive_0009  (447 frames, city turns + stops)"),
]

fig = plt.figure(figsize=(22, 11))
fig.suptitle(
    "ESKF GPS-Outage Stress Test -- KITTI 2011_09_26\n"
    f"Grey band = GPS blackout (30-55% of seq)  |  RPE window = {RPE_WINDOW} frames (~1 s)",
    fontsize=13, fontweight="bold",
)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.34)

for row, (fname, title) in enumerate(SEQS):
    p = Path(fname)
    if not p.exists():
        print(f"[warn] {fname} not found")
        continue

    d       = load_csv(p)
    frames  = d["frame"]
    outage  = d["outage"].astype(int)
    W       = RPE_WINDOW

    e_full  = ate(d["full_gps_x"],   d["full_gps_y"],   d["gt_x"], d["gt_y"])
    e_imu   = ate(d["imu_dead_x"],   d["imu_dead_y"],   d["gt_x"], d["gt_y"])
    e_aided = ate(d["aided_dead_x"], d["aided_dead_y"], d["gt_x"], d["gt_y"])

    r_full  = rpe_series(d["full_gps_x"],   d["full_gps_y"],   d["gt_x"], d["gt_y"])
    r_imu   = rpe_series(d["imu_dead_x"],   d["imu_dead_y"],   d["gt_x"], d["gt_y"])
    r_aided = rpe_series(d["aided_dead_x"], d["aided_dead_y"], d["gt_x"], d["gt_y"])
    rf      = frames[:len(r_full)]
    ro      = outage[:len(r_full)]

    pre_f, dur_f, post_f = phase_rmse(e_full,  outage)
    pre_i, dur_i, post_i = phase_rmse(e_imu,   outage)
    pre_a, dur_a, post_a = phase_rmse(e_aided, outage)
    peak_i = float(e_imu[outage == 1].max())   if outage.any() else 0
    peak_a = float(e_aided[outage == 1].max()) if outage.any() else 0

    # -- trajectory (cols 0-1) -------------------------------------------
    ax_t = fig.add_subplot(gs[row, 0:2])
    ax_t.plot(d["gt_x"],         d["gt_y"],         "k-",  lw=2.2, label="Ground truth",              zorder=5)
    ax_t.plot(d["full_gps_x"],   d["full_gps_y"],   "g-",  lw=1.5, label=f"Full GPS  RMSE={np.sqrt(np.mean(e_full**2)):.2f}m",  zorder=4)
    ax_t.plot(d["aided_dead_x"], d["aided_dead_y"], "b-",  lw=1.5, label=f"Aided DR  RMSE={np.sqrt(np.mean(e_aided**2)):.2f}m", zorder=3)
    ax_t.plot(d["imu_dead_x"],   d["imu_dead_y"],   "r--", lw=1.2, label=f"IMU only  RMSE={np.sqrt(np.mean(e_imu**2)):.2f}m",   zorder=2)
    for cx, cy, c in [("full_gps_x","full_gps_y","green"),
                       ("aided_dead_x","aided_dead_y","steelblue"),
                       ("imu_dead_x","imu_dead_y","salmon")]:
        m = outage.astype(bool)
        ax_t.scatter(d[cx][m], d[cy][m], s=5, c=c, alpha=0.35, zorder=2)
    ax_t.scatter([d["gt_x"][0]],  [d["gt_y"][0]],  s=80, c="limegreen", zorder=7)
    ax_t.scatter([d["gt_x"][-1]], [d["gt_y"][-1]], s=80, c="purple",    zorder=7)
    ax_t.annotate("Start", (d["gt_x"][0],  d["gt_y"][0]),  fontsize=7, ha="right")
    ax_t.annotate("End",   (d["gt_x"][-1], d["gt_y"][-1]), fontsize=7, ha="left")
    ax_t.set_title(title, fontsize=11)
    ax_t.set_xlabel("East (m)"); ax_t.set_ylabel("North (m)")
    ax_t.set_aspect("equal"); ax_t.legend(fontsize=7.5, loc="best")
    ax_t.grid(True, alpha=0.3)

    # -- ATE panel (col 2) -----------------------------------------------
    ax_a = fig.add_subplot(gs[row, 2])
    shade(ax_a, frames, outage)
    ax_a.plot(frames, e_full,  "g-",  lw=1.1, alpha=0.9,
              label=f"Full GPS  pre={pre_f:.1f} | out={dur_f:.1f} | post={post_f:.1f} m")
    ax_a.plot(frames, e_aided, "b-",  lw=1.1, alpha=0.9,
              label=f"Aided DR  pre={pre_a:.1f} | out={dur_a:.1f} | post={post_a:.1f} m")
    ax_a.plot(frames, e_imu,   "r--", lw=1.0, alpha=0.8,
              label=f"IMU only  pre={pre_i:.1f} | out={dur_i:.1f} | post={post_i:.1f} m")
    if outage.any():
        ax_a.annotate(f"peak {peak_i:.0f}m",
            xy=(frames[int(e_imu.argmax())], peak_i),
            xytext=(frames[int(e_imu.argmax())] - len(frames)*0.10, peak_i * 0.86),
            fontsize=7, color="red",
            arrowprops=dict(arrowstyle="-", color="red", lw=0.8))
        ax_a.annotate(f"peak {peak_a:.1f}m",
            xy=(frames[int(e_aided.argmax())], peak_a),
            xytext=(frames[int(e_aided.argmax())] + len(frames)*0.02, peak_a),
            fontsize=7, color="steelblue")
    ax_a.set_title("ATE -- per-frame position error", fontsize=10)
    ax_a.set_xlabel("Frame"); ax_a.set_ylabel("Error (m)")
    ax_a.legend(fontsize=6.5, loc="upper left"); ax_a.grid(True, alpha=0.3)

    # -- RPE panel (col 3) -----------------------------------------------
    ax_r = fig.add_subplot(gs[row, 3])
    shade(ax_r, rf, ro, label=None)
    ax_r.plot(rf, r_full,  "g-",  lw=1.1, alpha=0.9, label=f"Full GPS  mu={r_full.mean():.2f}m")
    ax_r.plot(rf, r_aided, "b-",  lw=1.1, alpha=0.9, label=f"Aided DR  mu={r_aided.mean():.2f}m")
    ax_r.plot(rf, r_imu,   "r--", lw=1.0, alpha=0.8, label=f"IMU only  mu={r_imu.mean():.2f}m")

    for r_s, c, tag in [(r_imu, "red", "IMU"), (r_aided, "steelblue", "Aided")]:
        tmp = r_s.copy(); tmp[ro == 0] = 0
        if tmp.max() > 0:
            pi = int(tmp.argmax())
            rate = tmp[pi] / (W * 0.1)
            ax_r.annotate(f"{tag} {rate:.1f}m/s",
                xy=(rf[pi], tmp[pi]),
                xytext=(rf[pi] - len(rf)*0.12, tmp[pi] * 0.82),
                fontsize=6.5, color=c,
                arrowprops=dict(arrowstyle="-", color=c, lw=0.7))

    ax_r.set_title(f"RPE -- {W}-frame relative error (~{W*0.1:.1f}s window)", fontsize=10)
    ax_r.set_xlabel("Frame (window start)"); ax_r.set_ylabel("RPE (m)")
    ax_r.legend(fontsize=6.5, loc="upper left"); ax_r.grid(True, alpha=0.3)

out = Path("eskf_results.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out.resolve()}")
plt.show()
