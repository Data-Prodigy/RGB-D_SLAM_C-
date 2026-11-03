import cv2
import numpy as np
import freenect
import os
import time

# Set your target directory to match what your C++ slam.exe expects (e.g., "C:/dev/")
OUTPUT_DIR = "C:/dev"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 50)
print("KINECT RGB-D SLAM DATASET RECORDER")
print(f"Saving frames to: {OUTPUT_DIR}")
print("=" * 50)
print("Press SPACE to save a single synced RGB+Depth pair.")
print("Press 'r' to toggle continuous recording on/off.")
print("Press ESC to exit.")
print("=" * 50)

frame_idx = 0
recording = False
last_saved_time = 0
RECORD_INTERVAL_S = 0.033  # ~30 FPS throttle for continuous recording (adjust as needed)


def get_rgb_frame():
    data, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    if data is not None:
        return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return None


def get_depth_frame():
    # Returns 16-bit depth values directly in MILLIMETERS
    data, _ = freenect.sync_get_depth(0, freenect.DEPTH_MM)
    return data


def depth_to_display_turbo(depth_mm):
    """Visualizes millimeter depth maps (0.5m - 3.5m) using Google's TURBO colormap."""
    min_mm = 500.0   # 0.5 meters (near clip)
    max_mm = 3500.0  # 3.5 meters (far clip)
    
    clipped = np.clip(depth_mm.astype(np.float32), min_mm, max_mm)
    scaled = ((clipped - min_mm) / (max_mm - min_mm) * 255.0).astype(np.uint8)
    colorized = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    
    invalid_mask = (depth_mm <= 0) | (depth_mm > 10000)
    colorized[invalid_mask] = [0, 0, 0]
    
    return colorized


def save_pair(rgb, depth, idx):
    # Match exact C++ naming convention: frame_XXXX.png (depth) and frame_XXXX_rgb.png (rgb)
    depth_path = os.path.join(OUTPUT_DIR, f"frame_{idx:04d}.png")
    rgb_path = os.path.join(OUTPUT_DIR, f"frame_{idx:04d}_rgb.png")
    
    # Save depth as 16-bit uint16 PNG (millimetric depth Z)
    cv2.imwrite(depth_path, depth.astype(np.uint16))
    # Save RGB as standard 8-bit BGR image
    cv2.imwrite(rgb_path, rgb)
    
    return rgb_path, depth_path


while True:
    rgb = get_rgb_frame()
    depth = get_depth_frame()

    if rgb is None or depth is None:
        continue

    # 1. Colorize depth map for real-time preview
    depth_vis = depth_to_display_turbo(depth)

    # 2. Combine feeds side-by-side for live display
    combined_view = np.hstack((rgb, depth_vis))

    # 3. Status overlay
    status = "RECORDING" if recording else "IDLE"
    status_color = (0, 0, 255) if recording else (0, 255, 0)
    
    cv2.putText(combined_view, f"RGB [{status}] - Frame {frame_idx:04d}", 
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(combined_view, "Depth MM (Preview)", 
                (655, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Kinect SLAM Recorder [RGB | Depth]", combined_view)

    key = cv2.waitKey(1) & 0xFF

    # Handle continuous recording timer
    if recording:
        now = time.time()
        if now - last_saved_time >= RECORD_INTERVAL_S:
            save_pair(rgb, depth, frame_idx)
            valid_depths = depth[depth > 0]
            min_d = valid_depths.min() if valid_depths.size > 0 else 0
            print(f"Recorded frame {frame_idx:04d} (MM range: min={min_d}mm, max={depth.max()}mm)")
            frame_idx += 1
            last_saved_time = now

    if key == 32:  # SPACE: single snapshot
        save_pair(rgb, depth, frame_idx)
        valid_depths = depth[depth > 0]
        min_d = valid_depths.min() if valid_depths.size > 0 else 0
        print(f"Captured snapshot {frame_idx:04d} (min={min_d}mm, max={depth.max()}mm)")
        frame_idx += 1

    elif key == ord('r'):  # Toggle recording
        recording = not recording
        print("Recording STARTED" if recording else "Recording STOPPED")

    elif key == 27:  # ESC to exit
        print("\nExiting capture loop...")
        break

freenect.sync_stop()
cv2.destroyAllWindows()