import numpy as np
import cv2
import mrcal
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RGB_MODEL_PATH   = "/home/uduakdon/kinect_cal/camera-RGB.cameramodel"
IR_MODEL_PATH    = "/home/uduakdon/kinect_cal/camera_IR.cameramodel"
CORNERS_RGB_PATH = "/home/uduakdon/kinect_cal/corners_stereo_rgb.vnl"
CORNERS_IR_PATH  = "/home/uduakdon/kinect_cal/corners_stereo_ir.vnl"
OBJECT_WIDTH_N   = 14          # corners per side (matches --object-width-n)
OBJECT_SPACING_M = 0.041       # meters between corners (matches --object-spacing)
OUTPUT_PATH      = "rgb_ir_extrinsics.txt"


# ---------------------------------------------------------------------------
# Reader: Parse vnlog file into dictionary keyed by base filename (frame_XXXX.png)
# ---------------------------------------------------------------------------
def read_corners_vnl(path):
    frames = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if parts[1] == "-":
                continue
            # Strip path so key is strictly "frame_0000.png"
            filename = Path(parts[0]).name
            x, y = float(parts[1]), float(parts[2])
            frames.setdefault(filename, []).append((x, y))
    return frames


# ---------------------------------------------------------------------------
# Board's local 3D coordinates (flat, z=0)
# ---------------------------------------------------------------------------
def board_object_points(n, spacing):
    pts = np.zeros((n * n, 3), dtype=np.float64)
    idx = 0
    for row in range(n):
        for col in range(n):
            pts[idx] = (col * spacing, row * spacing, 0.0)
            idx += 1
    return pts


# ---------------------------------------------------------------------------
# Unproject via mrcal model -> ideal unit-pinhole normalized coordinates
# ---------------------------------------------------------------------------
def undistort_to_normalized(pixel_xy, model):
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64)
    rays = mrcal.unproject(pixel_xy, *model.intrinsics())
    normalized = rays[:, :2] / rays[:, 2:3]
    return normalized


# ---------------------------------------------------------------------------
# Solve PnP in normalized space
# ---------------------------------------------------------------------------
def solve_board_pose(object_pts, normalized_xy):
    K_identity = np.eye(3)
    dist_zero = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(
        object_pts.astype(np.float64),
        normalized_xy.astype(np.float64),
        K_identity, dist_zero,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


# ---------------------------------------------------------------------------
# Kabsch / Umeyama rigid registration (SVD)
# ---------------------------------------------------------------------------
def kabsch_umeyama(P_src, P_dst):
    centroid_src = P_src.mean(axis=0)
    centroid_dst = P_dst.mean(axis=0)
    Q_src = P_src - centroid_src
    Q_dst = P_dst - centroid_dst

    H = Q_src.T @ Q_dst
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = centroid_dst - R @ centroid_src
    return R, t


def main():
    model_rgb = mrcal.cameramodel(RGB_MODEL_PATH)
    model_ir  = mrcal.cameramodel(IR_MODEL_PATH)

    frames_rgb = read_corners_vnl(CORNERS_RGB_PATH)
    frames_ir  = read_corners_vnl(CORNERS_IR_PATH)

    obj_pts = board_object_points(OBJECT_WIDTH_N, OBJECT_SPACING_M)

    # Explicit pair index mapping based on my capture logs 
    explicit_index_pairs = [
       #(0, 1),   # RGB 0  <-> IR 1
       (3,2),   # RGB 3  <-> IR 2
       #(4, 5),   # RGB 4  <-> IR 5
       #(7, 6),   # RGB 7  <-> IR 6
       #(8, 9),   # RGB 8  <-> IR 9
        (11, 10), # RGB 11 <-> IR 10
      #(12, 13), # RGB 12 <-> IR 13
      #(15, 14)  # RGB 15 <-> IR 14
    ]

    matched_pairs = []
    for rgb_idx, ir_idx in explicit_index_pairs:
        rgb_file = f"frame_{rgb_idx:04d}.png"
        ir_file  = f"frame_{ir_idx:04d}.png"

        if rgb_file in frames_rgb and ir_file in frames_ir:
            matched_pairs.append((rgb_file, ir_file))
        else:
            print(f"Warning: Pair missing corner detections ({rgb_file} or {ir_file})")

    if not matched_pairs:
        raise RuntimeError("No matched RGB/IR frame pairs found with valid detections.")

    print(f"Matched {len(matched_pairs)} RGB/IR frame pairs.")

    all_P_ir  = []
    all_P_rgb = []
    valid_rgb_pixels = []

    expected_corners = OBJECT_WIDTH_N ** 2

    for rgb_file, ir_file in matched_pairs:
        corners_rgb = frames_rgb[rgb_file]
        corners_ir  = frames_ir[ir_file]

        if len(corners_rgb) != expected_corners or len(corners_ir) != expected_corners:
            print(f"  skipping pair ({rgb_file} <-> {ir_file}): incomplete detection "
                  f"(rgb={len(corners_rgb)}, ir={len(corners_ir)})")
            continue

        norm_rgb = undistort_to_normalized(corners_rgb, model_rgb)
        norm_ir  = undistort_to_normalized(corners_ir, model_ir)

        pose_rgb = solve_board_pose(obj_pts, norm_rgb)
        pose_ir  = solve_board_pose(obj_pts, norm_ir)

        if pose_rgb is None or pose_ir is None:
            print(f"  skipping pair ({rgb_file} <-> {ir_file}): PnP failed")
            continue

        R_rgb, t_rgb = pose_rgb
        R_ir, t_ir   = pose_ir
        R_i = R_rgb @ R_ir.T
        t_i = t_rgb - R_i @ t_ir
        ang = np.degrees(np.arccos(np.clip((np.trace(R_i) - 1) / 2, -1, 1)))
        print(f"{rgb_file}<->{ir_file}: rot {ang:.2f} deg, t(cm) {np.round(t_i*100, 2)}")

        P_rgb = (R_rgb @ obj_pts.T).T + t_rgb
        P_ir  = (R_ir  @ obj_pts.T).T + t_ir

        all_P_rgb.append(P_rgb)
        all_P_ir.append(P_ir)
        valid_rgb_pixels.append(corners_rgb)

    if not all_P_rgb:
        raise RuntimeError("No pairs passed PnP and corner verification checks.")

    P_ir_all  = np.vstack(all_P_ir)
    P_rgb_all = np.vstack(all_P_rgb)
    all_rgb_pixels = np.vstack(valid_rgb_pixels)

    print(f"Pooled {P_ir_all.shape[0]} corresponding 3D points across all valid pairs.")

    R_ir2rgb, t_ir2rgb = kabsch_umeyama(P_ir_all, P_rgb_all)

    # Verify reprojection error
    P_rgb_predicted = np.ascontiguousarray((R_ir2rgb @ P_ir_all.T).T + t_ir2rgb)
    pixels_predicted = mrcal.project(P_rgb_predicted, *model_rgb.intrinsics())

    errors = np.linalg.norm(pixels_predicted - all_rgb_pixels, axis=1)
    print(f"\nReprojection error (RGB pixels): "
          f"mean={errors.mean():.3f}  median={np.median(errors):.3f}  max={errors.max():.3f}")

    print("\nR (IR -> RGB):\n", R_ir2rgb)
    print("t (IR -> RGB), meters:", t_ir2rgb)

    with open(OUTPUT_PATH, "w") as f:
        f.write("# Rotation matrix R (IR -> RGB frame)\n")
        for row in R_ir2rgb:
            f.write(" ".join(f"{v:.10f}" for v in row) + "\n")
        f.write("# Translation t (IR -> RGB frame), meters\n")
        f.write(" ".join(f"{v:.10f}" for v in t_ir2rgb) + "\n")
    print(f"\nWrote {OUTPUT_PATH}")

if __name__ == "__main__":
    main()