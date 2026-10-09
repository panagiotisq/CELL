#!/usr/bin/env python3
"""
Generate per-frame local LiDAR maps required by LEAR's DatasetM3ED.

For each pose index the global point cloud is transformed into the LiDAR frame,
cropped to the visible volume, then re-expressed in the event-camera frame and
saved as point_cloud_XXXXX.h5.

Also creates symlinks so the LEAR dataset root has the expected naming
(<lear_data_root>/<sequence_name>/):
    <sequence>_data.h5    → symlink to raw M3ED file
    <sequence>_pose_gt.h5 → symlink to raw M3ED file

Usage (run from anywhere):
    python3 generate_local_maps.py \
        --sequences falcon_flight_1 falcon_flight_2 falcon_flight_3 \
        --scene_type indoor

Defaults point to the paths already present on this machine.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm

try:
    import hdf5plugin
except ImportError:
    pass

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not found. Install with: pip install open3d")
    sys.exit(1)


M3ED_ROOT   = "/path/to/M3ED"   # EDIT: raw M3ED sequences (<seq>_data.h5, <seq>_pose_gt.h5)
OUTPUT_ROOT = "data/m3ed"       # EDIT: CELL data root to populate

# Per-scene bounding box (in LiDAR local frame) used to crop the global map.
# Keeps only points within a volume centred on the robot; reduces map size.
SCENE_BOUNDS = {
    "indoor":  {"x": (-1.0,  10.0), "y": (-5.0,   5.0)},
    "outdoor": {"x": (-1.0, 100.0), "y": (-25.0, 25.0)},
}


def load_global_pcd(pcd_path: Path, device: torch.device) -> torch.Tensor:
    """Read .pcd file → homogeneous point matrix [4, N]."""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)
    ones = torch.ones(pts.shape[0], 1, dtype=torch.float32, device=device)
    return torch.cat([pts, ones], dim=1).t()   # [4, N]


def find_file(seq_dir: Path, suffix: str, exclude: str = "") -> Path:
    """Return first .h5 / .pcd file in seq_dir matching suffix (excluding exclude)."""
    hits = [f for f in seq_dir.glob(f"*{suffix}") if not exclude or exclude not in f.name]
    if not hits:
        raise FileNotFoundError(f"No file matching *{suffix} in {seq_dir}")
    return sorted(hits)[0]


def process_sequence(seq_dir: Path, output_root: Path, bounds: dict, device: torch.device,
                     world_frame: bool = False):
    """Generate per-frame local maps.

    world_frame=False (default, unchanged): each map is expressed in the GT
        event-camera frame of that pose index (used by the registration pipeline).
    world_frame=True (pose-tracking / Option C): the SAME visibility-cropped points
        are stored in the fixed world frame (= LiDAR-frame-0, the global-map frame),
        so a running pose estimate can project them without any per-frame GT. Output
        goes to a separate 'local_maps_world/' dir; the original 'local_maps/' is
        never touched.
    """
    data_path = find_file(seq_dir, "_data.h5", exclude="pose_gt")
    pose_path = find_file(seq_dir, "_pose_gt.h5")
    pcd_path  = find_file(seq_dir, ".pcd")

    # Strip "_data" to get the canonical LEAR sequence name
    seq_name = data_path.stem.replace("_data", "")

    maps_subdir = "local_maps_world" if world_frame else "local_maps"
    out_maps_dir = output_root / seq_name / maps_subdir
    out_maps_dir.mkdir(parents=True, exist_ok=True)

    # Symlinks so LEAR's DatasetM3ED can find the raw H5 files under <lear_root>/<seq_name>/
    for src in [data_path, pose_path, pcd_path]:
        dst = output_root / seq_name / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())
            print(f"    symlink: {dst.name} → {src.resolve()}")

    print(f"  Loading global PCD ({pcd_path.name}) ...")
    global_map = load_global_pcd(pcd_path, device)
    print(f"  {global_map.shape[1]:,} points in global map")

    with h5py.File(data_path, "r") as f:
        # 4×4 rigid transform: LiDAR → Prophesee-left event camera
        T_lidar_to_ev = torch.tensor(
            f["/ouster/calib/T_to_prophesee_left"][:],
            dtype=torch.float32, device=device,
        )

    with h5py.File(pose_path, "r") as f:
        Ln_T_L0 = f["Ln_T_L0"][:]      # [N, 4, 4]  world → LiDAR-frame-n
        n_frames = Ln_T_L0.shape[0]

    xlo, xhi = bounds["x"]
    ylo, yhi = bounds["y"]

    existing = sum(1 for i in range(n_frames - 1) if (out_maps_dir / f"point_cloud_{i:05d}.h5").exists())
    if existing == n_frames - 1:
        print(f"  [SKIP] {seq_name}: {existing} maps already exist")
        return

    print(f"  Generating {n_frames - 1} local maps (scene bounds x∈[{xlo},{xhi}], y∈[{ylo},{yhi}]) ...")
    for idx in tqdm.tqdm(range(n_frames - 1)):
        out_file = out_maps_dir / f"point_cloud_{idx:05d}.h5"
        if out_file.exists():
            continue

        pose = torch.tensor(Ln_T_L0[idx], dtype=torch.float32, device=device)

        # Global → LiDAR-frame-n  (used to compute the visibility crop)
        local = torch.matmul(pose, global_map)            # [4, N]

        # Crop to bounding box (in LiDAR-frame-n)
        mask = (
            (local[0] > xlo) & (local[0] < xhi) &
            (local[1] > ylo) & (local[1] < yhi)
        )

        if world_frame:
            # Store the SAME visible points in the fixed world frame (no per-frame
            # GT transform baked in). A running pose estimate P (world->event-cam)
            # projects these via  P @ pc_world  at track time.
            # NOTE: world coordinates can be far from the origin on long trajectories
            # (car big_loop spans ~260 m), where float16 would quantize to ~13 cm.
            # So world maps are stored in float32; the GT-frame path keeps half.
            out_pc = global_map[:, mask].cpu().float()    # [4, M] world coords, fp32
        else:
            local = local[:, mask]
            # LiDAR-frame-n → event-camera frame (points stay near the camera, so
            # half precision is plenty here — unchanged original behaviour).
            out_pc = torch.matmul(T_lidar_to_ev, local).cpu().half()   # [4, M] GT-cam-frame-n

        with h5py.File(out_file, "w") as hf:
            hf.create_dataset("PC", data=out_pc, compression="lzf", shuffle=True)

    print(f"  Done → {out_maps_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate per-frame local LiDAR maps for LEAR.")
    parser.add_argument("--m3ed_root",   default=M3ED_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT,
                        help="LEAR dataset root. Output goes to <output_root>/<sequence_name>/local_maps/")
    parser.add_argument("--sequences", nargs="+",
                        default=["falcon_flight_1", "falcon_flight_2", "falcon_flight_3"],
                        help="Subdirectory names under m3ed_root.")
    parser.add_argument("--scene_type", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--world_frame", action="store_true",
                        help="Store maps in the fixed world frame (pose-tracking / Option C) "
                             "under local_maps_world/. Default off = original GT-frame local_maps/.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    m3ed_root   = Path(args.m3ed_root)
    output_root = Path(args.output_root)
    bounds      = SCENE_BOUNDS[args.scene_type]
    device      = torch.device(args.device)

    for folder in args.sequences:
        seq_dir = m3ed_root / folder
        if not seq_dir.exists():
            print(f"[WARN] {seq_dir} not found, skipping.")
            continue
        try:
            find_file(seq_dir, "_data.h5",   exclude="pose_gt")
            find_file(seq_dir, "_pose_gt.h5")
            find_file(seq_dir, ".pcd")
        except FileNotFoundError as e:
            print(f"[WARN] {folder}: {e} — skipping (incomplete data).")
            continue

        print(f"\nProcessing: {folder}{'  [world frame]' if args.world_frame else ''}")
        process_sequence(seq_dir, output_root, bounds, device, world_frame=args.world_frame)

    print(f"\nAll done.  LEAR data root: {output_root}")
    print("Pass this path as --dataset to LEAR's main.py")


if __name__ == "__main__":
    main()
