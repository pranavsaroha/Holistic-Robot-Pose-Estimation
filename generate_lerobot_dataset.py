#!/usr/bin/env python
import argparse, json, math, os, shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# ---------- helpers ----------------------------------------------------------
def compute_stats(arr: np.ndarray):
    """Compute statistics for an array. For multi-dimensional arrays, compute per dimension."""
    if arr.ndim == 1:
        return {
            "min": arr.min().tolist(),
            "max": arr.max().tolist(),
            "mean": arr.mean().tolist(),
            "std": arr.std(ddof=0).tolist(),
        }
    else:
        # For multi-dimensional arrays (like joints), compute per dimension
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0, ddof=0).tolist(),
        }

def compute_image_stats(video_path: Path, max_frames: int = 100):
    """Compute image statistics from video frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    # Sample frames for statistics (to avoid processing entire video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = min(max_frames, total_frames)
    step = max(1, total_frames // sample_frames)
    
    # Collect pixel values from sampled frames
    r_values, g_values, b_values = [], [], []
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and normalize to [0, 1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            r_values.extend(frame_rgb[:, :, 0].flatten())
            g_values.extend(frame_rgb[:, :, 1].flatten())
            b_values.extend(frame_rgb[:, :, 2].flatten())
    
    cap.release()
    
    # Compute statistics for each channel
    r_stats = compute_stats(np.array(r_values))
    g_stats = compute_stats(np.array(g_values))
    b_stats = compute_stats(np.array(b_values))
    
    return {
        "min": [[r_stats["min"]], [g_stats["min"]], [b_stats["min"]]],
        "max": [[r_stats["max"]], [g_stats["max"]], [b_stats["max"]]],
        "mean": [[r_stats["mean"]], [g_stats["mean"]], [b_stats["mean"]]],
        "std": [[r_stats["std"]], [g_stats["std"]], [b_stats["std"]]],
        "count": [total_frames]
    }

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

# ---------- main -------------------------------------------------------------
def main(args):
    vid_path = Path(args.video).expanduser()
    joint_path = Path(args.joints).expanduser()
    out_root = Path(args.out_root).expanduser()
    epi_idx = 0       # single‑episode dataset

    # --- prep output tree ----------------------------------------------------
    data_dir   = ensure_dir(out_root / "data" / "chunk-000")
    cam_dir    = ensure_dir(out_root / "videos" / "chunk-000" / "observation.images.main")
    meta_dir   = ensure_dir(out_root / "meta")

    # --- copy video ----------------------------------------------------------
    vid_dst = cam_dir / f"episode_{epi_idx:06d}.mp4"
    shutil.copy2(vid_path, vid_dst)

    # --- video metadata ------------------------------------------------------
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {vid_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    nfrm  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # --- load joints  --------------------------------------------------------
    joints = np.loadtxt(joint_path, delimiter=",", ndmin=2)
    if joints.shape[0] != nfrm:
        raise ValueError(
            f"Frame mismatch: video has {nfrm} frames, joints file {joints.shape[0]}"
        )
    n_joints = joints.shape[1]

    # --- build parquet table -------------------------------------------------
    action = np.vstack([joints[1:], joints[-1]])   # last action = hold
    df = pd.DataFrame({
        "observation.state": joints.tolist(),
        "action": action.tolist(),
        "timestamp": np.arange(nfrm) / fps,
        "episode_index": epi_idx,
        "frame_index": np.arange(nfrm),
        "index": np.arange(nfrm),          # global == local (single episode)
        "task_index": np.zeros(nfrm, dtype=np.int64),  # single task per episode
        "next.done": [False]*(nfrm-1) + [True],
    })
    pq.write_table(pa.Table.from_pandas(df),
                   data_dir / f"episode_{epi_idx:06d}.parquet",
                   compression="zstd")

    # --- empty tasks file ----------------------------------------------------
    (meta_dir / "tasks.jsonl").touch()

    # --- episodes.jsonl ------------------------------------------------------
    episode_data = {
        "episode_index": epi_idx,
        "task_index": 0,
        "start_frame": 0,
        "end_frame": nfrm - 1,
        "num_frames": nfrm,
        "duration": nfrm / fps,
        "video_path": f"videos/chunk-000/observation.images.main/episode_{epi_idx:06d}.mp4",
        "split": "train"
    }
    
    with open(meta_dir / "episodes.jsonl", "w") as f:
        f.write(json.dumps(episode_data) + "\n")

    # --- episodes_stats.jsonl ------------------------------------------------
    # Compute statistics for all data fields
    joint_stats = compute_stats(joints)
    action_stats = compute_stats(action)
    
    # Add count field to all stats
    joint_stats["count"] = [nfrm]
    action_stats["count"] = [nfrm]
    
    # Compute statistics for other fields
    timestamps = np.arange(nfrm) / fps
    frame_indices = np.arange(nfrm)
    episode_indices = np.full(nfrm, epi_idx)
    global_indices = np.arange(nfrm)
    task_indices = np.zeros(nfrm, dtype=np.int64)
    
    timestamp_stats = compute_stats(timestamps)
    frame_index_stats = compute_stats(frame_indices)
    episode_index_stats = compute_stats(episode_indices)
    index_stats = compute_stats(global_indices)
    task_index_stats = compute_stats(task_indices)
    
    # Add count field to all stats
    timestamp_stats["count"] = [nfrm]
    frame_index_stats["count"] = [nfrm]
    episode_index_stats["count"] = [nfrm]
    index_stats["count"] = [nfrm]
    task_index_stats["count"] = [nfrm]
    
    # Compute real image statistics with proper count
    image_stats = compute_image_stats(vid_path, max_frames=100)
    
    episode_stats = {
        "episode_index": epi_idx,
        "stats": {
            "action": action_stats,
            "observation.state": joint_stats,
            "observation.images.main": image_stats,
            "timestamp": timestamp_stats,
            "frame_index": frame_index_stats,
            "episode_index": episode_index_stats,
            "index": index_stats,
            "task_index": task_index_stats
        }
    }
    
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        f.write(json.dumps(episode_stats) + "\n")

    # --- dataset card --------------------------------------------------------
    dataset_card = f"""---
language:
- en
license: mit
multimodality:
- video
paperswithcode_id: null
pretty_name: {args.robot_type} Teleop Dataset
tags:
- robotics
- imitation-learning
- teleoperation
- video
task_categories:
- robotics
task_ids:
- robotics-manipulation
---

# Dataset Card for {args.robot_type} Teleop Dataset

## Dataset Description

- **Repository:** [Add repository URL]
- **Paper:** [Add paper reference if applicable]
- **Point of Contact:** [Add contact information]

### Dataset Summary

This dataset contains teleoperation data for {args.robot_type} robot manipulation tasks. The dataset includes:

- **Videos:** RGB video recordings of robot manipulation
- **Joint States:** Per-frame joint angle data
- **Actions:** Robot joint actions derived from state differences
- **Metadata:** Episode and frame indexing information

### Supported Tasks and Leaderboards

This dataset is designed for robot imitation learning and manipulation tasks.

### Languages

The dataset contains English language metadata.

## Dataset Structure

### Data Instances

Each episode contains:
- Video frames at {fps} FPS
- Joint state data for {n_joints} joints
- Action data for robot control
- Timestamp information
- Episode and frame indexing

### Data Fields

- `observation.state`: Joint angle data (float32, shape: [{n_joints}])
- `action`: Robot joint actions (float32, shape: [{n_joints}])
- `observation.images.main`: RGB video data (video, shape: [{height}, {width}, 3])
- `frame_index`: Frame index within episode (int64)
- `episode_index`: Episode identifier (int64)
- `index`: Global frame index (int64)
- `task_index`: Task identifier (int64)
- `timestamp`: Time from episode start (float32)
- `next.done`: Episode termination flag (bool)

### Data Splits

- **Train:** {nfrm} frames across 1 episode

## Dataset Creation

### Source Data

#### Initial Data Collection and Normalization

The dataset was created from teleoperation recordings of {args.robot_type} robot manipulation tasks.

#### Who are the source language producers?

[Add information about data collection process]

### Annotations

#### Annotation process

[Add information about annotation process if applicable]

#### Who are the annotators?

[Add information about annotators if applicable]

### Personal and Sensitive Information

[Add information about personal/sensitive data handling]

## Additional Information

### Dataset Curators

[Add curator information]

### Licensing Information

This dataset is licensed under the MIT License.

### Citation Information

```bibtex
@dataset{{{args.robot_type}_teleop_dataset,
  title = {{{args.robot_type} Teleop Dataset}},
  author = {{[Add author information]}},
  year = {{2024}},
  url = {{[Add dataset URL]}}
}}
```

### Contributions

[Add contribution information]

### Contact

[Add contact information]
"""
    
    with open(out_root / "README.md", "w") as f:
        f.write(dataset_card)

    # --- info.json -----------------------------------------------------------
    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "fps": fps,
        "total_episodes": 1,
        "total_frames": nfrm,
        "total_tasks": 0,
        "total_videos": 1,
        "splits": {"train": "0:1"},
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [n_joints],
                "names": [f"joint_{i}" for i in range(n_joints)],
            },
            "action": {
                "dtype": "float32",
                "shape": [n_joints],
                "names": [f"joint_{i}" for i in range(n_joints)],
            },
            "observation.images.main": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "info": {"video.fps": fps, "video.codec": "mp4v"},
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "next.done": {
                "dtype": "bool",
                "shape": [1],
                "names": None,
            },
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # --- dataset config ------------------------------------------------------
    config = {
        "builder_name": "lerobot",
        "config_name": args.robot_type,
        "version": {"version_str": "0.0.0"},
        "data_dir": str(out_root),
        "download_checksums": {},
        "features": info["features"],
        "supervised_keys": None,
        "disable_tqdm": False,
        "disable_nullable_warning": False,
    }
    
    with open(out_root / "dataset_info.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Dataset written to: {out_root.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build a LeRobot v2.1 dataset from one teleop video + joint angles")
    p.add_argument("--video", required=True, help="Path to MP4")
    p.add_argument("--joints", required=True, help="CSV with per-frame joint angles (frames × joints)")
    p.add_argument("--out_root", default="my_dataset", help="Output dataset folder")
    p.add_argument("--robot_type", default="custom_robot", help="Anything, e.g., so100")
    main(p.parse_args())
