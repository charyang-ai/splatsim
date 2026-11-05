#!/usr/bin/env python
"""Make the two CUDA extensions build with hipcc. Idempotent; run from the repo root.

Every change is guarded by __HIP_PLATFORM_AMD__ or is valid CUDA as well, so a
patched tree still builds with nvcc.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DGR = ROOT / "submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/diff-gaussian-rasterization"
KNN = ROOT / "submodules/simple-knn"

MARK = "__HIP_PLATFORM_AMD__"

REDUCE_OLD = "#include <cooperative_groups/reduce.h>\n"
REDUCE_NEW = (
    "#if !defined(__HIP_PLATFORM_AMD__)\n"
    "#include <cooperative_groups/reduce.h>\n"
    "#endif\n"
)

LAUNCH_PARAMS_OLD = '#include "device_launch_parameters.h"\n'
LAUNCH_PARAMS_NEW = (
    "#if !defined(__HIP_PLATFORM_AMD__)\n"
    '#include "device_launch_parameters.h"\n'
    "#endif\n"
)

# (file, [(old, new), ...]). Each old string must occur exactly once unless it is a launch fix.
PATCHES = {
    DGR / "cuda_rasterizer/rasterizer_impl.cu": [
        (
            '#include "device_launch_parameters.h"\n'
            "#include <cub/cub.cuh>\n"
            "#include <cub/device/device_radix_sort.cuh>\n",
            "#if defined(__HIP_PLATFORM_AMD__)\n"
            "#include <hipcub/hipcub.hpp>\n"
            "namespace cub = hipcub;\n"
            "#else\n"
            '#include "device_launch_parameters.h"\n'
            "#include <cub/cub.cuh>\n"
            "#include <cub/device/device_radix_sort.cuh>\n"
            "#endif\n",
        ),
        (REDUCE_OLD, REDUCE_NEW),
    ],
    DGR / "cuda_rasterizer/forward.cu": [(REDUCE_OLD, REDUCE_NEW)],
    DGR / "cuda_rasterizer/backward.cu": [(REDUCE_OLD, REDUCE_NEW)],
    DGR / "cuda_rasterizer/forward.h": [(LAUNCH_PARAMS_OLD, LAUNCH_PARAMS_NEW)],
    DGR / "cuda_rasterizer/backward.h": [(LAUNCH_PARAMS_OLD, LAUNCH_PARAMS_NEW)],
    DGR / "cuda_rasterizer/auxiliary.h": [
        (
            "\t\t\t__trap();\n",
            "#if defined(__HIP_PLATFORM_AMD__)\n"
            "\t\t\t__builtin_trap();\n"
            "#else\n"
            "\t\t\t__trap();\n"
            "#endif\n",
        ),
    ],
    KNN / "simple_knn.cu": [
        (
            '#include "device_launch_parameters.h"\n'
            '#include "simple_knn.h"\n'
            "#include <cub/cub.cuh>\n"
            "#include <cub/device/device_radix_sort.cuh>\n",
            '#include "simple_knn.h"\n'
            "#if defined(__HIP_PLATFORM_AMD__)\n"
            "#include <hipcub/hipcub.hpp>\n"
            "namespace cub = hipcub;\n"
            "#else\n"
            '#include "device_launch_parameters.h"\n'
            "#include <cub/cub.cuh>\n"
            "#include <cub/device/device_radix_sort.cuh>\n"
            "#endif\n",
        ),
        (
            "#define __CUDACC__\n#include <cooperative_groups.h>\n",
            "#if !defined(__HIP_PLATFORM_AMD__)\n#define __CUDACC__\n#endif\n#include <cooperative_groups.h>\n",
        ),
        (REDUCE_OLD, REDUCE_NEW),
    ],
}

# nvcc accepts "<< <" and ">> >" around launch configurations; clang does not.
LAUNCH_FILES = [
    DGR / "cuda_rasterizer/rasterizer_impl.cu",
    DGR / "cuda_rasterizer/forward.cu",
    DGR / "cuda_rasterizer/backward.cu",
    KNN / "simple_knn.cu",
]


def main() -> int:
    for path, edits in PATCHES.items():
        text = path.read_text()
        if MARK in text:
            print(f"already patched  {path.relative_to(ROOT)}")
            continue
        for old, new in edits:
            count = text.count(old)
            if count != 1:
                raise SystemExit(f"{path.relative_to(ROOT)}: expected one match, found {count}:\n{old}")
            text = text.replace(old, new)
        path.write_text(text, newline="\n")
        print(f"patched          {path.relative_to(ROOT)}")

    for path in LAUNCH_FILES:
        text = path.read_text()
        new = text.replace("<< <", "<<<").replace(">> >", ">>>")
        if new != text:
            path.write_text(new, newline="\n")
            print(f"launch syntax    {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
