#!/usr/bin/env python
"""Numerical checks of the two hipcc-built extensions, no data needed.

simple-knn: distCUDA2 returns, per point, the mean squared distance to its three
nearest neighbours. Its box pruning never drops a box that could hold a closer
point, so it should match brute force up to float rounding.

diff-gaussian-rasterization: a random scene is rendered twice (the forward pass
has no atomics, so the two images must be identical) and pushed through the
backward pass, whose gradients must be finite.

    python radeon/smoke_kernels.py [--out results/smoke_kernels.json]
"""
import argparse
import json
import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from simple_knn._C import distCUDA2


def knn_check(n: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(0)
    pts = torch.rand(n, 3, device="cuda", generator=g) * torch.tensor([4.0, 2.0, 1.0], device="cuda")
    got = distCUDA2(pts)
    torch.cuda.synchronize()
    best = torch.empty(n, 3, device="cuda")
    for i in range(0, n, 1024):
        block = pts[i:i + 1024]
        d = (block[:, None, :] - pts[None, :, :]).pow(2).sum(-1)  # exact, no matmul shortcut
        d[torch.arange(len(block)), torch.arange(i, i + len(block))] = float("inf")
        best[i:i + 1024] = d.topk(3, largest=False).values
    ref = best.mean(1)
    rel = (got - ref).abs() / ref.clamp_min(1e-12)
    return {"points": n, "max_rel_err": float(rel.max()), "mean_rel_err": float(rel.mean())}


def look_at(eye, target):
    f = torch.nn.functional.normalize(target - eye, dim=0)
    r = torch.nn.functional.normalize(torch.linalg.cross(f, torch.tensor([0.0, 0.0, 1.0])), dim=0)
    u = torch.linalg.cross(r, f)
    w2c = torch.eye(4)
    w2c[0, :3], w2c[1, :3], w2c[2, :3] = r, -u, f
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


def raster_check(n: int, width: int, height: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(1)
    fov = math.radians(60)
    tanx = math.tan(fov / 2)
    tany = tanx * height / width
    near, far = 0.01, 100.0
    proj = torch.zeros(4, 4)
    proj[0, 0], proj[1, 1] = 1 / tanx, 1 / tany
    proj[2, 2], proj[2, 3], proj[3, 2] = far / (far - near), -far * near / (far - near), 1.0
    eye = torch.tensor([3.0, -3.0, 2.0])
    w2c = look_at(eye, torch.zeros(3))
    view = w2c.T.contiguous().cuda()             # the rasterizer takes row-vector (transposed) matrices
    full = (proj @ w2c).T.contiguous().cuda()
    settings = GaussianRasterizationSettings(
        image_height=height, image_width=width, tanfovx=tanx, tanfovy=tany,
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0, viewmatrix=view, projmatrix=full,
        sh_degree=3, campos=eye.cuda(), prefiltered=False, debug=False, antialiasing=False)
    raster = GaussianRasterizer(settings)

    means = ((torch.rand(n, 3, device="cuda", generator=g) - 0.5) * 2).requires_grad_(True)
    shs = (torch.randn(n, 16, 3, device="cuda", generator=g) * 0.3).requires_grad_(True)
    opac = torch.sigmoid(torch.randn(n, 1, device="cuda", generator=g)).requires_grad_(True)
    scales = (torch.rand(n, 3, device="cuda", generator=g) * 0.03 + 0.005).requires_grad_(True)
    rots = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda", generator=g), dim=1).requires_grad_(True)
    means2d = torch.zeros_like(means, requires_grad=True)

    def forward():
        return raster(means3D=means, means2D=means2d, opacities=opac, shs=shs, scales=scales, rotations=rots)

    img1, radii, _ = forward()
    img2, _, _ = forward()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        forward()
    end.record()
    torch.cuda.synchronize()
    (img1 * torch.randn_like(img1)).sum().backward()
    grads = [t.grad for t in (means, shs, opac, scales, rots, means2d)]
    return {
        "gaussians": n, "size": [width, height],
        "visible": int((radii > 0).sum()),
        "image_mean": float(img1.mean()), "image_max": float(img1.max()),
        "forward_identical": bool(torch.equal(img1, img2)),
        "forward_ms": round(start.elapsed_time(end) / 10, 2),
        "grads_finite": bool(all(torch.isfinite(gr).all() for gr in grads)),
        "grad_abs_mean_means3D": float(means.grad.abs().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    result = {
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "hip": torch.version.hip,
        "simple_knn": knn_check(20000),
        "rasterizer": raster_check(200000, 1280, 720),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
