#!/usr/bin/env python3
"""
Prepare TUM RGB-D freiburg1_room for my direct RGB-D odometry program.

What it does
  1. Associates every depth image with the nearest RGB image in time (<= 20 ms apart).
  2. UNDISTORTS both images with the published freiburg1 RGB intrinsics + distortion.
     (The raw TUM images are NOT rectified; fr1 has strong distortion, k1 = 0.26, k3 = 1.16.)
       - RGB   : bilinear
       - depth : nearest neighbour (never interpolate depth: it invents values at object edges)
     TUM depth is already registered to the RGB image, so the same maps serve both.
  3. Writes frame_0000_rgb.png / frame_0000.png, ... with NO gaps in the numbering
     (the C++ program stops at the first missing frame).
  4. Writes camera_params.yaml in the same OpenCV-FileStorage layout as the Kinect one.
  6. Writes frame_times.txt (frame index -> real TUM timestamps). main.cpp writes the frame INDEX as the
     trajectory timestamp; run fix_trajectory_timestamps.py before evo so poses pair with the right
     ground-truth poses.
  5. Writes ir_map.bin and rgb_map.bin as IDENTITY maps in the layout my camera.hpp reads
     (int32 w, int32 h, 4 x double fx fy cx cy, then w*h float32 pairs (x, y)).
     The images are already undistorted, so "map every pixel to itself" is correct, and the
     loader stays strict (no fallback needed). 

Usage
  python prepare_tum.py
  python prepare_tum.py --dataset C:/dev/rgbd_dataset_freiburg1_room --target C:/dev/tum_ready
"""

import argparse
import os
import struct
import sys

import cv2
import numpy as np

#  TUM freiburg1 RGB intrinsics (depth is registered to this camera) 
W, H = 640, 480
FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3
DIST = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633], dtype=np.float64)  # k1 k2 p1 p2 k3
DEPTH_SCALE = 5000.0          # 16-bit PNG: 5000 units = 1 metre
MAX_TIME_DIFF = 0.02          # seconds


def load_file_list(dataset_dir, filename):
    path = os.path.join(dataset_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {filename} in {dataset_dir}")
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            data.append((float(parts[0]), parts[1]))
    return data


def associate(rgb_list, depth_list, max_diff):
    """For each depth image, the nearest RGB image in time (both lists are time-sorted).
    Returns (rgb_time, rgb_path, depth_time, depth_path) tuples."""
    matches = []
    i = 0
    for d_time, d_path in depth_list:
        while i < len(rgb_list) - 1 and abs(rgb_list[i][0] - d_time) > abs(rgb_list[i + 1][0] - d_time):
            i += 1
        if abs(rgb_list[i][0] - d_time) < max_diff:
            matches.append((rgb_list[i][0], rgb_list[i][1], d_time, d_path))
    return matches


# camera_params.yaml (OpenCV FileStorage layout, same keys as the Kinect file) 
def _mat(rows, cols, data, indent):
    body = ", ".join(repr(float(v)) for v in data)
    pad = " " * indent
    return (f"!!opencv-matrix\n{pad}   rows: {rows}\n{pad}   cols: {cols}\n"
            f"{pad}   dt: d\n{pad}   data: [ {body} ]")


def _camera_block(name, map_path):
    k = _mat(3, 3, [FX, 0, CX, 0, FY, CY, 0, 0, 1], 9)
    return f"""{name}:
   raw_image_width_px: {W}
   raw_image_height_px: {H}
   lens_model: "LENSMODEL_PINHOLE"
   calibrated_imager_width_px: {W}
   calibrated_imager_height_px: {H}
   central_fx_px: {FX!r}
   central_fy_px: {FY!r}
   central_cx_px: {CX!r}
   central_cy_px: {CY!r}
   lens_model_parameter_count: 0
   lens_model_parameters: [ ]
   has_opencv_distortion_coefficients: 0
   is_spline_lens_model: 0
   extrinsics_rt_fromref_raw: [ 0., 0., 0., 0., 0., 0. ]
   undistorted_pinhole:
      map_file: "{map_path}"
      width_px: {W}
      height_px: {H}
      fx_px: {FX!r}
      fy_px: {FY!r}
      cx_px: {CX!r}
      cy_px: {CY!r}
      camera_matrix_K: {k}
      map_data_embedded_in_this_file: 0
"""


def camera_params_yaml(target_dir):
    t = target_dir.replace("\\", "/")
    ident3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    ident4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    return (
        "%YAML:1.0\n---\n"
        'description: "TUM freiburg1_desk. Depth is registered to RGB, so IR and RGB are one pinhole camera. '
        'Images are undistorted by prepare_tum.py (zero residual distortion)."\n'
        "depth:\n"
        f"   unit_to_meters: {1.0 / DEPTH_SCALE!r}\n"
        f"   raw_units_per_meter: {DEPTH_SCALE!r}\n"
        f"   image_width_px: {W}\n"
        f"   image_height_px: {H}\n"
        + _camera_block("ir_camera", f"{t}/ir_map.bin")      # these files do not exist: identity map
        + _camera_block("rgb_camera", f"{t}/rgb_map.bin")
        + "ir_to_rgb_extrinsics:\n"
        f"   rotation_R_rgb_from_ir: {_mat(3, 3, ident3, 3)}\n"
        f"   translation_t_rgb_from_ir_m: {_mat(3, 1, [0, 0, 0], 3)}\n"
        f"   transform_T_rgb_from_ir_4x4: {_mat(4, 4, ident4, 3)}\n"
    )


def write_identity_map(path):
    """Binary map in the layout of camera.hpp::loadUndistortMap: every pixel maps to itself."""
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    m = np.ascontiguousarray(np.dstack([xs, ys]).astype("<f4"))      # (H, W, 2) = CV_32FC2, x then y
    with open(path, "wb") as f:
        f.write(struct.pack("<ii4d", W, H, FX, FY, CX, CY))
        f.write(m.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="C:/dev/rgbd_dataset_freiburg1_room")
    ap.add_argument("--target", default="C:/dev/tum_ready1")
    a = ap.parse_args()

    os.makedirs(a.target, exist_ok=True)

    print("Loading rgb.txt and depth.txt...")
    rgb_list = load_file_list(a.dataset, "rgb.txt")
    depth_list = load_file_list(a.dataset, "depth.txt")
    matches = associate(rgb_list, depth_list, MAX_TIME_DIFF)
    print(f"Matched {len(matches)} frame pairs (<= {MAX_TIME_DIFF * 1000:.0f} ms apart).")

    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(K, DIST, None, K, (W, H), cv2.CV_32FC1)

    written = 0
    times = []                       # (frame index, rgb timestamp, depth timestamp) of every frame written
    for rgb_t, rgb_rel, depth_t, depth_rel in matches:
        rgb = cv2.imread(os.path.join(a.dataset, rgb_rel), cv2.IMREAD_COLOR)
        depth = cv2.imread(os.path.join(a.dataset, depth_rel), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            print(f"  skipped (unreadable): {rgb_rel} / {depth_rel}")
            continue
        if rgb.shape[:2] != (H, W) or depth.shape[:2] != (H, W) or depth.dtype != np.uint16:
            print(f"  skipped (unexpected size/dtype): {rgb_rel} / {depth_rel}")
            continue

        rgb_u = cv2.remap(rgb, map1, map2, cv2.INTER_LINEAR)
        # nearest neighbour for depth; pixels that fall outside the source become 0 = invalid
        depth_u = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        cv2.imwrite(os.path.join(a.target, f"frame_{written:04d}_rgb.png"), rgb_u)
        cv2.imwrite(os.path.join(a.target, f"frame_{written:04d}.png"), depth_u)
        times.append((written, rgb_t, depth_t))
        written += 1
        if written % 100 == 0:
            print(f"  {written} frames written...")

    # frame_XXXX is NOT the TUM timestamp: keep the real times so the trajectory can be evaluated.
    with open(os.path.join(a.target, "frame_times.txt"), "w", newline="\n") as f:
        f.write("# frame_index rgb_timestamp depth_timestamp\n")
        for idx, rt, dt in times:
            f.write(f"{idx} {rt:.6f} {dt:.6f}\n")

    yaml_path = os.path.join(a.target, "camera_params.yaml")
    # LF newlines, UTF-8 without BOM: OpenCV's YAML reader is picky about the first bytes of the file.
    with open(yaml_path, "w", newline="\n", encoding="utf-8") as f:
        f.write(camera_params_yaml(a.target))
    with open(yaml_path, "rb") as f:
        head = f.read(12)
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    readable = fs.isOpened() and not fs.getNode("ir_camera").getNode("undistorted_pinhole").empty()
    fs.release()
    print(f"camera_params.yaml: {os.path.getsize(yaml_path)} bytes, starts with {head!r}, "
          f"OpenCV read-back {'OK' if readable else 'FAILED'}")
    if not readable:
        sys.exit("camera_params.yaml was written but OpenCV cannot read it back.")

    for name in ("ir_map.bin", "rgb_map.bin"):
        write_identity_map(os.path.join(a.target, name))

    print(f"Preparation complete: {written} frames + camera_params.yaml + identity maps in {a.target}")
    if written == 0:
        sys.exit("No frames were written. Check --dataset.")


if __name__ == "__main__":
    main()