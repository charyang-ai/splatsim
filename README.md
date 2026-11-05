# SplatSim on one AMD Radeon Pro W7900

上游是 [qureshinomaan/SplatSim](https://github.com/qureshinomaan/SplatSim)，`main` @ `ea07395d`。论文是 [arXiv:2409.10161](https://arxiv.org/abs/2409.10161)（ICRA 2025），项目页 [splatsim.github.io](https://splatsim.github.io/)。上游说明原样保留在 [`UPSTREAM_README.md`](UPSTREAM_README.md)。

这次在一张 AMD Radeon Pro W7900D 上跑完了上游 README 的渲染主路径：

```bash
python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name robot_iphone
python scripts/run_env_sim.py --agent replay_trajectory_and_save
```

上游发布的 420 条苹果放盘子轨迹全部回放完：37,241 个时间步，每步一张 800×449 的高斯溅射渲染图，用时 95 分钟，显存 4.4 GiB。另外在同一张卡上重画了作者训练视角的渲染，和作者随数据发布的渲染图逐像素比，六个视角的 PSNR 都在 96 dB 以上，最大差一个灰度级。

## 许可

SplatSim 的代码是 GPL-2.0。仓库主人 2024-10-08 的初始提交在根目录建了 `LICENSE`（GPL-2.0），2025-08 的重构（#12）把它和一批旧文件一起挪进了 `to_delete/`，两处是同一个 blob `d159169d`。这里把它放回根目录。

子模块各有许可：gaussian-splatting、diff-gaussian-rasterization 和 simple-knn 是 Inria 的 Gaussian-Splatting License，只限非商业研究和评估；GELLO 和 pybullet-URDF-models 是 MIT；ghalton 是 LGPL-3.0。这个仓库只记录子模块的提交，不分发它们的代码；对子模块的改动由 `radeon/patch_submodules.py` 在本地打上。

## 改了什么

SplatSim 自己的 Python 代码没有改。要动的是两个 CUDA 扩展、一个失效的子模块地址和两份写死作者家目录的配置。

| 文件 | 改了什么 |
| --- | --- |
| `radeon/patch_submodules.py` | 让 diff-gaussian-rasterization（`9c5c2028`，`dr_aa` 分支）和 simple-knn（`60f461f4`）能用 hipcc 编译。7 个文件，+40/−12，全部用 `__HIP_PLATFORM_AMD__` 守住，或者本身就是合法 CUDA，原样还能用 nvcc 编译。补丁内容见 [`results/submodules.diff`](results/submodules.diff) |
| `.gitmodules` | `c-yiting/pybullet-URDF-models` 已经 404，换成 `GPT-RL/pybullet-URDF-models`，里面有同一个提交 `ca8f7951` |
| `configs/object_configs/objects.yaml`、`configs/folder_configs.yaml` | `/home/jennyw2/data/` 改成仓库内的 `./data/` |
| `LICENSE` | 从 `to_delete/` 放回根目录 |
| `to_delete/` | 去掉。约 780 MB 的旧副本，不在任何运行路径上 |
| `.gitignore` | 忽略下载的 `data/test_data/`、`data/output/`、`data/bc_data/`；`results/` 不受上游 `*.png`、`*logs*` 规则影响 |
| `radeon/` | 容器、依赖、数据、检查和回放脚本，见下文 |

补丁里的六处改动：

- CUB 换成 hipCUB，`namespace cub = hipcub`，调用处一个字不动。
- `<< <` 和 `>> >` 改成 `<<<`、`>>>`。nvcc 容忍启动配置里的空格，clang 不容忍。
- `__trap()` 在 HIP 下换成 `__builtin_trap()`。
- `cooperative_groups/reduce.h` 在 HIP 下不包含。这两个扩展只用到 `this_grid()` 和 `this_thread_block()`，它们在主头文件里。
- `device_launch_parameters.h` 在 HIP 下不包含，`threadIdx` 这些是 HIP 的内建量。
- simple-knn 在包含 cooperative groups 之前自己 `#define __CUDACC__`，HIP 下不定义。

GLM 不用改：固定的 `5c46b9c`（0.9.9.9）认得 `__HIP__`，函数自动带上 `__host__ __device__`。

第一个提交 `0b88952` 是上游 `ea07395d` 去掉 `to_delete/`（保留 LICENSE）后的树，171 个条目逐个 blob 和上游相同，`git diff 0b88952` 看 SplatSim 层面的全部改动。

## 环境

- 镜像：`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`，PyTorch `2.11.0+rocm7.14.0`，HIP 7.14.60850。
- GPU：一张 AMD Radeon Pro W7900D（gfx1100，48 GB）。容器只挂 `/dev/kfd` 和一个渲染节点，容器内 `HIP_VISIBLE_DEVICES=0`、`ROCR_VISIBLE_DEVICES=0`，`torch.cuda.device_count()` 是 1。
- hipCUB / rocPRIM / rocThrust：镜像里 pip 装的 ROCm 不带这三个纯头文件库。`radeon/fetch_rocm_headers.sh` 从 repo.radeon.com 解出 ROCm 7.2.4 的 4.2.0 `-dev` 包，`install.sh` 通过 `CPATH` 交给 hipcc。wheel 里只有 `libamdhip64.so.7`，链接用的 `libamdhip64.so` 由 `install.sh` 建一个符号链接。
- 两个扩展必须 `--no-build-isolation` 编译，否则 pip 会在隔离环境里装一个 CUDA 版 torch。
- PyBullet 3.2.7 在 PyPI 上只有源码包，在容器里编译。`opencv-python==4.12.0.88` 要求 `numpy<2.3`，镜像里的 numpy 2.5.1 被换成 2.2.6，torch 不受影响。其余依赖见 [`results/pip_freeze.txt`](results/pip_freeze.txt)。
- 显示：上游假定有显示器（PyBullet `p.GUI` 和 `cv2.imshow`）。这里用 Xvfb，PyBullet 的 GUI 由 Mesa llvmpipe 在 CPU 上画，高斯渲染在 W7900 上。

## 复现

**1. 代码。** 上游的 `--recursive` 也能用，只是会把 SIBR viewer 一起拉下来。回放用到的子模块是这些：

```bash
git clone https://github.com/charyang-ai/splatsim.git src && cd src
git submodule update --init submodules/gaussian-splatting-wrapper/gaussian_splatting submodules/simple-knn \
  submodules/gello_software submodules/ghalton submodules/pybullet-URDF-models \
  submodules/pybullet-playground-wrapper/pybullet_playground submodules/pybullet_ur5_gripper
git -C submodules/gaussian-splatting-wrapper/gaussian_splatting submodule update --init --recursive \
  submodules/diff-gaussian-rasterization
```

**2. 数据。** 上游 README 链接的三个 Google Drive 压缩包一共 60 GB。`radeon/fetch_data.py` 用 HTTP range 请求读压缩包的中央目录，只解出 `robot_iphone` 场景、苹果、盘子等物体的 splat 和几张作者的渲染图；76 MB 的轨迹包整个下载。一共约 1.5 GB，放进 `data/`：

```bash
python radeon/fetch_data.py
```

**3. 容器。** 工作目录挂到 `/workspace`，仓库放在 `$WORK/src`：

```bash
bash $WORK/src/radeon/start_container.sh $WORK /dev/dri/renderD131
docker exec radeon-splatsim bash /workspace/src/radeon/fetch_rocm_headers.sh /workspace/rocm-headers
docker exec -e ROCM_HEADERS=/workspace/rocm-headers/include radeon-splatsim bash /workspace/src/radeon/install.sh
```

`install.sh` 还认 `PIP_INDEX_URL` 和 `APT_MIRROR`（例如 `http://mirrors.aliyun.com/ubuntu/`）。

**4. 检查和回放。**

```bash
docker exec -w /workspace/src radeon-splatsim python radeon/smoke_kernels.py --out results/smoke_kernels.json
docker exec -w /workspace/src radeon-splatsim python radeon/check_render.py --views 0 3 120 240 360 480
docker exec -w /workspace/src radeon-splatsim bash radeon/replay.sh results/replay
docker exec -w /workspace/src radeon-splatsim python radeon/summarize_replay.py --out results/replay
```

`replay.sh` 先起 Xvfb，再按上游 README 依次起服务端和回放端。上游的回放端放完最后一条轨迹后会一直空转，脚本看到 `No more recorded trajectories.` 就把两边停掉。图像写在 `data/bc_data/gello/<轨迹>/images_1/`。

## 验证

### 两个扩展

[`results/smoke_kernels.json`](results/smoke_kernels.json)：

- simple-knn：两万个随机点，`distCUDA2` 和逐点暴力求最近三个邻居的结果比，最大相对误差 2.6×10⁻⁷。
- 光栅器：二十万个随机高斯、1280×720，同一场景画两次结果逐位相同，一次前向 1.94 ms，反向梯度全部有限。

### 和作者的渲染图比

作者用 CUDA 版 3DGS 的 `render.py` 画过全部 482 个训练视角，结果随 `output.zip` 发布。`radeon/check_render.py` 在 W7900 上用同一份 splat（2,192,636 个高斯）、同样的全分辨率 1907×1071 和黑背景重画，结果在 [`results/check_render/`](results/check_render/)。

| 视角 | 照片 | 和作者渲染的 PSNR | 最大差 | 和照片的 PSNR（这里 / 作者） | 渲染耗时 |
| --- | --- | --- | --- | --- | --- |
| 0 | `frame_00001.png` | 96.18 dB | 1 | 27.52 / 27.52 dB | 19.6 ms |
| 3 | `frame_00004.png` | 96.87 dB | 1 | 27.96 / 27.96 dB | 20.7 ms |
| 120 | `frame_00121.png` | 97.37 dB | 1 | 29.16 / 29.16 dB | 15.4 ms |
| 240 | `frame_00271.png` | 97.31 dB | 1 | 23.49 / 23.49 dB | 17.4 ms |
| 360 | `frame_00391.png` | 97.14 dB | 1 | 25.17 / 25.17 dB | 15.6 ms |
| 480 | `frame_00511.png` | 96.81 dB | 1 | 29.65 / 29.65 dB | 15.7 ms |

视角 3 是回放用的相机，2,042,397 个像素里有 82 个差一个灰度级，其余逐位相同。四联图见 [`view3_comparison.jpg`](results/check_render/view3_comparison.jpg)：W7900、作者的渲染、照片和两次渲染之差。

### 上游主路径

[`results/replay/replay_summary.json`](results/replay/replay_summary.json)、[`per_trajectory.tsv`](results/replay/per_trajectory.tsv)：

- 420 条轨迹全部回放，37,241 张图，和录下的 37,241 个时间步一一对应（每条 49 到 172 步），没有一条缺帧。
- 每张 800×449：COLMAP 照片先被缩到 1.6K 宽，`cam_scale=2` 再减半。
- 2026-11-04 15:24:19 到 16:59:32（UTC），第一张到最后一张 5,708.5 秒，6.52 张/秒，相邻两张的间隔中位数 0.152 秒。
- GPU 3 的占用中位数 53%，最高 60%，显存 4.4 GiB（[`results/gpu_samples.txt`](results/gpu_samples.txt)，每 30 秒一次）。
- 抽样的六条轨迹，图像平均灰度 145–148，相邻两帧平均变化 1.4–1.7 个灰度级：画面不是空的，也不是静止的。[`contact_sheet.jpg`](results/replay/contact_sheet.jpg) 是这六条轨迹的首、三分之一、三分之二和末帧，[`trajectory_0.mp4`](results/replay/trajectory_0.mp4) 是第一条。
- [`display.png`](results/replay/display.png) 是回放进行到第 2 分钟时 Xvfb 的截图：前面是 `cv2.imshow` 的窗口，后面是 PyBullet 的 GUI。
- 盘子画出来是一块黑色圆盘。发布的盘子 splat 里，所有高斯的基色都在截断线以下，这是数据本身的样子。

### 时间花在哪

回放进行到第 8 分钟时用 py-spy 不暂停地各采了 30 秒（[`results/profile/`](results/profile/)）。服务端处理 `get_observations` 的线程里，`prep_image_rendering` 占全部样本的 36%，其中 `transform_means` 30%、球谐旋转 `transform_shs` 23%（e3nn 的 `wigner_D` 14%），`render` 只有 3.6%。回放端 84% 的样本在等服务端回 `get_observations`。每帧 0.15 秒里，大头是在 CPU 上按 PyBullet 的 20 个关节逐个构造 Wigner-D 矩阵、旋转每段高斯，不是光栅化。

### 没有验证的

- 论文的 Diffusion Policy 训练和真机部署不在这个仓库里，上游 README 写着 “TODO for diffusion policy”。86.25% 的真机成功率不在这次运行的范围内。
- 运动规划器生成示教（`sim_ur_pybullet_apple`）和 GELLO 遥操作没有跑，回放用的是上游发布的轨迹。
- 服务器访问不了 Google Drive，数据在别处下载后拷过去。482 张 COLMAP 照片只拷了用到的六张：加载器只打开选中视角的照片。
