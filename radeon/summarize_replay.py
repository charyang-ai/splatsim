#!/usr/bin/env python
"""Summarise a finished replay: frames per trajectory, throughput, and a few sanity checks.

    python radeon/summarize_replay.py --out results/replay
"""
import argparse
import json
import os
import subprocess

import numpy as np
import yaml
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-folder", default=None, help="defaults to traj_folder in configs/folder_configs.yaml")
    ap.add_argument("--out", default="results/replay")
    ap.add_argument("--sheet-trajectories", type=int, default=6)
    args = ap.parse_args()

    traj_folder = args.traj_folder
    if traj_folder is None:
        with open("configs/folder_configs.yaml") as f:
            traj_folder = yaml.safe_load(f)["traj_folder"]
    os.makedirs(args.out, exist_ok=True)

    order = sorted(os.listdir(traj_folder))  # the order ReplayTrajectoryAgent uses
    per_traj, stamps = [], []
    for name in order:
        folder = os.path.join(traj_folder, name)
        if not os.path.isdir(folder):
            continue
        steps = sorted(f for f in os.listdir(folder) if f.endswith(".pkl"))
        img_dir = os.path.join(folder, "images_1")
        frames = sorted(f for f in os.listdir(img_dir) if f.endswith(".png")) if os.path.isdir(img_dir) else []
        times = [os.path.getmtime(os.path.join(img_dir, f)) for f in frames]
        stamps.extend(times)
        per_traj.append({"trajectory": name, "steps": len(steps), "frames": len(frames),
                         "first": min(times) if times else None, "last": max(times) if times else None})

    stamps = np.sort(np.array(stamps))
    gaps = np.diff(stamps)
    replayed = [t for t in per_traj if t["frames"] > 0]
    mismatched = [t for t in replayed if t["frames"] != t["steps"]]

    sample_dirs = [os.path.join(traj_folder, t["trajectory"], "images_1") for t in replayed[: args.sheet_trajectories]]
    pixel_stats, motion = [], []
    rows = []
    for d in sample_dirs:
        frames = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        imgs = [np.asarray(Image.open(os.path.join(d, f)).convert("RGB")) for f in frames]
        pixel_stats.append([float(np.mean(i)) for i in imgs[:: max(1, len(imgs) // 8)]])
        motion.append(float(np.mean([np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
                                     for a, b in zip(imgs[:-1], imgs[1:])])))
        picks = [imgs[int(round(q * (len(imgs) - 1)))] for q in (0.0, 0.33, 0.66, 1.0)]
        rows.append(np.concatenate(picks, axis=1))
    if rows:
        sheet = Image.fromarray(np.concatenate(rows, axis=0))
        sheet.thumbnail((2400, 2400))
        sheet.save(os.path.join(args.out, "contact_sheet.jpg"), quality=90)
        first = sample_dirs[0]
        name = os.listdir(first)[0].rsplit("_", 1)[0]
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "15",
                        "-i", os.path.join(first, f"{name}_%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        os.path.join(args.out, f"trajectory_{replayed[0]['trajectory']}.mp4")], check=False)

    with Image.open(os.path.join(sample_dirs[0], sorted(os.listdir(sample_dirs[0]))[0])) as im:
        size = list(im.size)
    summary = {
        "trajectories_total": len(per_traj),
        "trajectories_replayed": len(replayed),
        "steps_total": int(sum(t["steps"] for t in per_traj)),
        "frames_total": int(sum(t["frames"] for t in per_traj)),
        "trajectories_frames_ne_steps": [{k: t[k] for k in ("trajectory", "steps", "frames")} for t in mismatched],
        "frame_size": size,
        "wall_seconds_first_to_last_frame": round(float(stamps[-1] - stamps[0]), 1) if len(stamps) else None,
        "frames_per_second": round(float((len(stamps) - 1) / (stamps[-1] - stamps[0])), 2) if len(stamps) > 1 else None,
        "frame_gap_seconds_median": round(float(np.median(gaps)), 4) if len(gaps) else None,
        "frame_gap_seconds_p90": round(float(np.percentile(gaps, 90)), 4) if len(gaps) else None,
        "sample_mean_pixel_values": pixel_stats,
        "sample_mean_abs_change_between_frames": motion,
    }
    with open(os.path.join(args.out, "replay_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "per_trajectory.tsv"), "w") as f:
        f.write("trajectory\tsteps\tframes\n")
        for t in per_traj:
            f.write(f"{t['trajectory']}\t{t['steps']}\t{t['frames']}\n")
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("sample_")}, indent=2))


if __name__ == "__main__":
    main()
