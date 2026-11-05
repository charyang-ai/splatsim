#!/usr/bin/env python
"""Render training views of the robot splat here and compare with the authors' renders.

The authors ran gaussian_splatting/render.py on their GPU and shipped the
results in output.zip (output/<robot>/train/ours_<iter>/{renders,gt}). This
script repeats that render for a few views on the current device and reports
PSNR against both their render and the ground-truth photo.

    python radeon/check_render.py --views 0 3 120 240 360 480 --out results/check_render
"""
import argparse
import json
import math
import os
from argparse import Namespace

import numpy as np
import torch
import torchvision
from PIL import Image

import gaussian_splatting  # noqa: F401  puts the fork's top-level modules on sys.path
from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.gaussian_renderer import GaussianModel, render
from gaussian_splatting.scene import Scene


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="robot_iphone")
    ap.add_argument("--data", default="data")
    ap.add_argument("--reference", default=None, help="defaults to <data>/output/<robot>/train/ours_<iter>")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--out", default="results/check_render")
    args = ap.parse_args()

    model_path = os.path.join(args.data, "output", args.robot)
    source_path = os.path.join(args.data, "test_data", args.robot)
    # Older cfg_args files predate the depths / train_test_exp options, so start from the fork's defaults.
    cfg = dict(sh_degree=3, images="images", depths="", resolution=-1, white_background=False,
               train_test_exp=False, eval=False)
    with open(os.path.join(model_path, "cfg_args")) as f:
        cfg.update(vars(eval(f.read(), {"Namespace": Namespace})))
    cfg.update(source_path=os.path.abspath(source_path), model_path=model_path, data_device="cuda")

    parser = argparse.ArgumentParser()
    dataset = ModelParams(parser, sentinel=True).extract(Namespace(**cfg))
    pipe = PipelineParams(parser).extract(Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                                                    debug=False, antialiasing=False))

    views = sorted(set(args.views))
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False,
                      train_cam_indices=views, test_cam_indices=[0])
        reference = args.reference or os.path.join(model_path, "train", f"ours_{scene.loaded_iter}")
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        os.makedirs(args.out, exist_ok=True)

        rows = []
        for idx, cam in zip(views, scene.getTrainCameras()):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            render(cam, gaussians, pipe, bg)  # warm-up
            start.record()
            img = render(cam, gaussians, pipe, bg)["render"]
            end.record()
            torch.cuda.synchronize()
            ours_path = os.path.join(args.out, f"{idx:05d}.png")
            torchvision.utils.save_image(img, ours_path)

            ours = np.asarray(Image.open(ours_path).convert("RGB"))
            theirs = np.asarray(Image.open(os.path.join(reference, "renders", f"{idx:05d}.png")).convert("RGB"))
            gt = np.asarray(Image.open(os.path.join(reference, "gt", f"{idx:05d}.png")).convert("RGB"))
            diff = np.abs(ours.astype(np.int16) - theirs.astype(np.int16))
            rows.append({
                "view": idx,
                "image_name": cam.image_name,
                "size": [int(ours.shape[1]), int(ours.shape[0])],
                "render_ms": round(start.elapsed_time(end), 2),
                "psnr_vs_their_render": round(psnr(ours, theirs), 2),
                "max_abs_diff": int(diff.max()),
                "pixels_off_by_more_than_2": round(float((diff.max(axis=2) > 2).mean()), 6),
                "psnr_ours_vs_gt": round(psnr(ours, gt), 2),
                "psnr_theirs_vs_gt": round(psnr(theirs, gt), 2),
            })
            print(json.dumps(rows[-1]))

    summary = {
        "robot": args.robot,
        "iteration": scene.loaded_iter,
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "views": rows,
    }
    with open(os.path.join(args.out, "check_render.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
