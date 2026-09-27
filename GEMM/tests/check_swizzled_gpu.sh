#!/usr/bin/env bash
# Run from the repository root on an SM80+ CUDA machine.
dir="drive/MyDrive/GPU_Accelerate_Compute/CUDA_GEMM_OPTIMIZATION/GEMM"
set -euo pipefail
nvcc -O3 -std=c++17 -lineinfo -arch=sm_80 -Xptxas=-v \
  $dir/gemm_tensor_core.cu -lcublas -o /tmp/gemm_swizzled
for shape in '1 1 1' '16 16 16' '32 32 48' '65 129 33' '129 65 129' '256 256 256'; do
  read -r m n k <<< "$shape"
  /tmp/gemm_swizzled "$m" "$n" "$k" 1
done
# Three K stages, including a partial final stage, exercise ping-pong reuse.
compute-sanitizer --error-exitcode 1 --tool memcheck /tmp/gemm_swizzled 129 145 129 1
compute-sanitizer --error-exitcode 1 --tool racecheck /tmp/gemm_swizzled 129 145 129 1
compute-sanitizer --error-exitcode 1 --tool synccheck /tmp/gemm_swizzled 129 145 129 1
