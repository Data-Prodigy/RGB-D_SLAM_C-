#!/usr/bin/env python3
"""
Turn an mrcal .cameramodel (e.g. LENSMODEL_SPLINED_STEREOGRAPHIC) into a
pinhole undistortion map that C++/OpenCV can use with cv::remap.
Creates an ideal virtual pinhole camera (LENSMODEL_PINHOLE) sharing the exact same resolution, intrinsic focal/center parameters, and extrinsics as the source camera.
This acts as the destination grid for undistorted images.

The output image is a pinhole image that keeps the model's own fx, fy, cx, cy,
so your existing pinhole math (unproject / project) becomes correct.

Usage:
    python3 make_undistort_maps.py camera_IR.cameramodel ir_map.bin  [--test frame_0000.png]
    python3 make_undistort_maps.py camera-RGB.cameramodel rgb_map.bin [--test frame_0000_rgb.png]

Binary format (little endian):
    int32  width
    int32  height
    float64 fx, fy, cx, cy      (intrinsics of the undistorted pinhole image)
    float32 map[height][width][2]   (x, y) = where to sample in the ORIGINAL image
"""
import argparse
import struct

import cv2
import numpy as np
import mrcal


def build_map(model_path):
    model = mrcal.cameramodel(model_path)
    W, H = (int(v) for v in model.imagersize())
    _lensmodel, intrinsics = model.intrinsics()
    fx, fy, cx, cy = (float(v) for v in intrinsics[:4])

    # Target model: ideal pinhole with the same central parameters.
    # Give it the SAME extrinsics as the source so the relative rotation is
    # identity no matter which mrcal version's default is in effect.
    kw = {}
    try:
        kw["extrinsics_rt_fromref"] = model.extrinsics_rt_fromref()
    except Exception:
        pass
    pinhole = mrcal.cameramodel(
        intrinsics=("LENSMODEL_PINHOLE", np.array([fx, fy, cx, cy])),
        imagersize=(W, H),
        **kw,
    )

    # For every pixel of the pinhole (destination) image, where do we sample
    # in the original (spline-distorted) image? Shape (H, W, 2), float32.
    m = mrcal.image_transformation_map(model, pinhole)
    m = np.ascontiguousarray(m, dtype=np.float32)
    assert m.shape == (H, W, 2), f"unexpected map shape {m.shape}"

    # Anything non-finite -> out of bounds, so cv::remap fills it with 0.
    m[~np.isfinite(m)] = -1.0
    return m, (W, H), (fx, fy, cx, cy)


def write_map(path, m, size, k):
    W, H = size
    with open(path, "wb") as f:
        f.write(struct.pack("<ii4d", W, H, *k))
        f.write(m.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cameramodel")
    ap.add_argument("out_bin")
    ap.add_argument("--test", help="optional image to undistort for a visual check")
    args = ap.parse_args()

    m, size, k = build_map(args.cameramodel)
    write_map(args.out_bin, m, size, k)
    print(f"wrote {args.out_bin}: {size[0]}x{size[1]}, "
          f"fx={k[0]:.3f} fy={k[1]:.3f} cx={k[2]:.3f} cy={k[3]:.3f}")

    if args.test:
        img = cv2.imread(args.test, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"could not read {args.test}")
        interp = cv2.INTER_NEAREST if img.dtype == np.uint16 else cv2.INTER_LINEAR
        out = cv2.remap(img, m, None, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out_path = args.out_bin.rsplit(".", 1)[0] + "_test.png"
        cv2.imwrite(out_path, out)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()