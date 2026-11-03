#!/usr/bin/env python3
"""
Accurate trajectory plots for a TUM-format RGB-D SLAM run.

What was wrong with the old script
----------------------------------
1. It paired poses by ARRAY POSITION (gt[:min_len] vs est[:min_len]).  TUM ground truth
   is ~100 Hz (~4900 rows for fr1/room) while the estimate has one pose per frame
   (~1360), so row i of one file is a different moment than row i of the other.
   Alignment and errors were computed on mismatched pairs.
   -> Now pairs by nearest TIMESTAMP (same as evo's --t_max_diff).
2. Raw overlay drew the estimate (first-camera frame, y points DOWN) on top of the
   ground truth (mocap world frame, z UP) in x-y.  Those frames are unrelated, so the
   picture meant nothing.  -> Raw view now draws each in its own top-down plane.
3. The error curve used Sim(3) while your evo number (0.117 m) is SE(3), so the plot
   and your reported RMSE disagreed.  -> Both are plotted, RMSE in the legend.
4. The x axis of the error plot was GT timestamps of the wrong rows.
   -> Seconds since start, from the matched pairs.

Usage
-----
python plot_trajectory.py --gt groundtruth.txt --est trajectory_tum_fixed.txt --out-dir plots
Optional: --max-diff 0.02  --offset 0.0 (seconds added to est timestamps)
"""
import argparse
import os
import sys

import numpy as np


# ----------------------------------------------------------------------------- IO
def load_tum(path):
    """Load a TUM trajectory: t x y z qx qy qz qw. Sorted by time."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.replace(",", " ").split()
            if len(p) < 4:
                continue
            rows.append([float(v) for v in p[:8]] if len(p) >= 8 else [float(v) for v in p[:4]])
    if not rows:
        raise ValueError(f"No poses read from {path}")
    arr = np.array(rows)
    return arr[np.argsort(arr[:, 0])]


# ----------------------------------------------------------------------- association
def associate(gt_t, est_t, max_diff):
    """For each estimate timestamp, nearest GT index within max_diff.
    Returns (est_idx, gt_idx) of matched pairs (each GT row used at most once)."""
    idx = np.searchsorted(gt_t, est_t)
    lo = np.clip(idx - 1, 0, len(gt_t) - 1)
    hi = np.clip(idx, 0, len(gt_t) - 1)
    pick = np.where(np.abs(gt_t[lo] - est_t) <= np.abs(gt_t[hi] - est_t), lo, hi)
    dt = np.abs(gt_t[pick] - est_t)
    ok = dt <= max_diff
    est_idx = np.nonzero(ok)[0]
    gt_idx = pick[ok]
    # keep the closest estimate when two share one GT row
    best = {}
    for e, g in zip(est_idx, gt_idx):
        if g not in best or dt[e] < dt[best[g]]:
            best[g] = e
    pairs = sorted((e, g) for g, e in best.items())
    return (np.array([p[0] for p in pairs], dtype=int),
            np.array([p[1] for p in pairs], dtype=int))


# ----------------------------------------------------------------------- alignment
def umeyama(src, dst, with_scale):
    """Least-squares R, t, s with dst ~= s * R @ src + t (Umeyama 1991, as in evo)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def apply(pts, R, t, s):
    return s * (pts @ R.T) + t


def stats(err):
    return dict(rmse=float(np.sqrt((err ** 2).mean())), mean=float(err.mean()),
                median=float(np.median(err)), max=float(err.max()),
                std=float(err.std()))


def path_length(xyz):
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def analyse(gt, est, max_diff, offset=0.0):
    est_t = est[:, 0] + offset
    e_idx, g_idx = associate(gt[:, 0], est_t, max_diff)
    if len(e_idx) < 3 or len(e_idx) < 0.1 * len(est):
        raise RuntimeError(
            f"Only {len(e_idx)} timestamp matches within {max_diff}s.\n"
            f"  GT time range : {gt[0,0]:.3f} .. {gt[-1,0]:.3f}\n"
            f"  EST time range: {est_t[0]:.3f} .. {est_t[-1]:.3f}\n"
            "If the estimate's times look like 0,1,2,... they are frame indices; run "
            "fix_trajectory_timestamps.py first. Wrong sequence's GT gives the same symptom.")
    g_xyz = gt[g_idx, 1:4]
    e_xyz = est[e_idx, 1:4]
    t = est_t[e_idx] - est_t[e_idx][0]
    out = dict(est_idx=e_idx, gt_idx=g_idx, t=t, gt_xyz=g_xyz, est_xyz=e_xyz,
               n_est=len(est), n_gt=len(gt), n_pairs=len(e_idx))
    for name, scale in (("se3", False), ("sim3", True)):
        R, tr, s = umeyama(e_xyz, g_xyz, scale)
        al = apply(e_xyz, R, tr, s)
        err = np.linalg.norm(g_xyz - al, axis=1)
        out[name] = dict(R=R, t=tr, s=s, aligned=al, err=err, **stats(err))
    return out


# --------------------------------------------------------------------------- plots
def make_plots(res, gt, est, out_dir, title_prefix):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(out_dir, exist_ok=True)
    GREEN, BLUE, RED = "#2e8b57", "#1f5fbf", "#c0392b"
    g, e = res["gt_xyz"], res["est_xyz"]
    written = []

    def save(fig, name):
        p = os.path.join(out_dir, name)
        fig.write_html(p)
        written.append(p)

    def markers(fig, xy, color, label, row=None, col=None):
        kw = dict(row=row, col=col) if row else {}
        fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers",
                                 marker=dict(symbol="circle", size=11, color=color,
                                             line=dict(color="white", width=1.5)),
                                 name=f"{label} start", showlegend=False), **kw)
        fig.add_trace(go.Scatter(x=[xy[-1, 0]], y=[xy[-1, 1]], mode="markers",
                                 marker=dict(symbol="square", size=11, color=color,
                                             line=dict(color="white", width=1.5)),
                                 name=f"{label} end", showlegend=False), **kw)

    # 1. Raw, each in its own frame (no alignment)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Ground truth (mocap frame) top-down: x vs y",
        "SLAM estimate (first-camera frame) top-down: x vs z"))
    fig.add_trace(go.Scatter(x=gt[:, 1], y=gt[:, 2], mode="lines", name="Ground truth",
                             line=dict(color=GREEN, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=est[:, 1], y=est[:, 3], mode="lines", name="SLAM estimate",
                             line=dict(color=BLUE, width=3)), row=1, col=2)
    markers(fig, gt[:, 1:3], GREEN, "GT", 1, 1)
    markers(fig, est[:, [1, 3]], BLUE, "SLAM", 1, 2)
    fig.update_xaxes(title_text="x (m)")
    fig.update_yaxes(title_text="y (m)", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_yaxes(title_text="z (m)", row=1, col=2, scaleanchor="x2", scaleratio=1)
    fig.update_layout(
        title=dict(text=f"{title_prefix}: raw trajectories (different frames; not comparable)", y=0.92),
        margin=dict(t=80, r=50, b=50, l=50)
    )
    save(fig, "plot_1_raw_own_frames.html")

    # 2 and 3. Aligned overlays in the GT frame (matched pairs), x vs y.
    for key, label, fname in (("se3", "SE(3) aligned", "plot_2_se3_aligned.html"),
                              ("sim3", "Sim(3) aligned", "plot_3_sim3_aligned.html")):
        r = res[key]
        a = r["aligned"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g[:, 0], y=g[:, 1], mode="lines", name="Ground truth",
                                 line=dict(color=GREEN, width=3)))
        fig.add_trace(go.Scatter(
            x=a[:, 0], y=a[:, 1], mode="lines+markers", name=f"SLAM ({label})",
            line=dict(color=BLUE, width=2), marker=dict(size=4, color=r["err"],
            colorscale="YlOrRd", cmin=0, cmax=max(r["max"], 1e-6),
            colorbar=dict(title="error (m)", x=1.05, len=0.8)),
            customdata=r["err"], hovertemplate="x=%{x:.3f} y=%{y:.3f}<br>err=%{customdata:.3f} m"))
        markers(fig, g[:, :2], GREEN, "GT")
        extra = f", scale={r['s']:.4f}" if key == "sim3" else ""
        fig.update_layout(
            title=dict(text=f"{title_prefix}: {label} | RMSE {r['rmse']:.4f} m, max {r['max']:.3f} m{extra} ({res['n_pairs']} matched poses)", y=0.92),
            xaxis_title="x (m)", yaxis_title="y (m)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(t=80, r=100, b=50, l=50)  # Added right padding for colorbar
        )
        save(fig, fname)

    # 4. Error vs time, both alignments.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["t"], y=res["se3"]["err"], mode="lines",
                             name=f"SE(3)  RMSE {res['se3']['rmse']:.4f} m",
                             line=dict(color=RED, width=2)))
    fig.add_trace(go.Scatter(x=res["t"], y=res["sim3"]["err"], mode="lines",
                             name=f"Sim(3) RMSE {res['sim3']['rmse']:.4f} m",
                             line=dict(color="#7f8c8d", width=2, dash="dash")))
    fig.update_layout(
        title=dict(text=f"{title_prefix}: absolute translation error over time", y=0.92),
        xaxis_title="time since start (s)", yaxis_title="error (m)",
        margin=dict(t=80, r=50, b=50, l=50)
    )
    save(fig, "plot_4_error_vs_time.html")

    # 5. Per-axis (aligned SE(3))
    a = res["se3"]["aligned"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("x (m)", "y (m)", "z (m)"))
    for i in range(3):
        fig.add_trace(go.Scatter(x=res["t"], y=g[:, i], mode="lines", name="Ground truth",
                                 line=dict(color=GREEN, width=2), showlegend=(i == 0)), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=res["t"], y=a[:, i], mode="lines", name="SLAM (SE(3))",
                                 line=dict(color=BLUE, width=2), showlegend=(i == 0)), row=i + 1, col=1)
    fig.update_xaxes(title_text="time since start (s)", row=3, col=1)
    fig.update_layout(
        title=dict(text=f"{title_prefix}: per-axis position (SE(3) aligned)", y=0.95),
        height=850,
        margin=dict(t=100, r=50, b=50, l=50)
    )
    save(fig, "plot_5_per_axis.html")

    # 6. 3-D view, SE(3) aligned, coloured by error.
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=g[:, 2], mode="lines", name="Ground truth",
                               line=dict(color=GREEN, width=5)))
    fig.add_trace(go.Scatter3d(x=a[:, 0], y=a[:, 1], z=a[:, 2], mode="lines+markers",
                               name="SLAM (SE(3))", line=dict(color=BLUE, width=3),
                               marker=dict(size=2.5, color=res["se3"]["err"], colorscale="YlOrRd",
                                           colorbar=dict(title="error (m)", x=1.1, len=0.8))))
    fig.update_layout(
        title=dict(text=f"{title_prefix}: 3-D, SE(3) aligned", y=0.92),
        scene=dict(aspectmode="data", xaxis_title="x (m)",
                   yaxis_title="y (m)", zaxis_title="z (m)"),
        margin=dict(t=80, r=100, b=50, l=50)
    )
    save(fig, "plot_6_3d.html")
    return written

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="TUM groundtruth.txt")
    ap.add_argument("--est", required=True, help="TUM-format estimate with REAL timestamps")
    ap.add_argument("--out-dir", default="plots")
    ap.add_argument("--max-diff", type=float, default=0.02,
                    help="max timestamp gap for a match, seconds (evo default 0.01)")
    ap.add_argument("--offset", type=float, default=0.0, help="seconds added to est timestamps")
    ap.add_argument("--title", default="RGB-D SLAM")
    args = ap.parse_args()

    gt = load_tum(args.gt)
    est = load_tum(args.est)
    if est.shape[1] < 4 or gt.shape[1] < 4:
        sys.exit("Files need at least: timestamp x y z")
    res = analyse(gt, est, args.max_diff, args.offset)

    print(f"GT poses {res['n_gt']}, estimated poses {res['n_est']}, matched {res['n_pairs']} "
          f"(max_diff {args.max_diff}s)")
    if res["n_pairs"] < 0.9 * res["n_est"]:
        print(f"  WARNING: only {100*res['n_pairs']/res['n_est']:.0f}% of estimated poses matched; "
              "RMSE covers just those. Try a larger --max-diff.")
    print(f"GT path length (matched poses): {path_length(res['gt_xyz']):.2f} m")
    for k, name in (("se3", "SE(3)"), ("sim3", "Sim(3)")):
        r = res[k]
        extra = f"  scale {r['s']:.4f}" if k == "sim3" else ""
        print(f"{name:7s} RMSE {r['rmse']:.4f}  mean {r['mean']:.4f}  median {r['median']:.4f}  "
              f"max {r['max']:.4f} m{extra}")
    files = make_plots(res, gt, est, args.out_dir, args.title)
    print("Wrote:\n  " + "\n  ".join(files))


if __name__ == "__main__":
    main()