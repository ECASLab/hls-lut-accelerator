# Non-Uniform Piecewise Linear Approximation Accelerator for FPGA-based Machine Learning Inference

This repository contains a hardware implementation of a **Non-Uniform Piecewise Linear (PWL) approximation accelerator** targeting FPGA devices using AMD/Xilinx Vitis HLS and Vitis toolchain.

This repository includes:
- Coefficient generation (Python)
- HLS synthesis and verification
- FPGA deployment targeting AMD Alveo U55C via Vitis `v++` flow

---

# 1. Authors

**Primary Author**
- Patrick Hugo Nepveu Nelson

**Supervision**
- Jorge Castro-Godínez
- Luis G. León-Vega

---

# 2. Scope

This repository supports the results presented in the associated thesis and conference submission under review.

It enables reproduction of:
- Fixed-point PWL coefficient generation
- Functional correctness via HLS simulation
- Hardware synthesis results (latency, II, resource usage)
- FPGA deployment binary generation (`.xclbin`)

---

# 3. Environment Requirements

## System Requirements

- Linux (tested on Ubuntu 22.04)
- Python ≥ 3.8
- AMD/Xilinx Vitis HLS
- AMD/Xilinx Vitis 2024.1 (or compatible)
- Xilinx Runtime (XRT)
- FPGA target: AMD Alveo U55C (required for full hardware reproduction)

---

## Toolchain Setup (HPC environment)

```bash
module load vivado/2024.1
source /opt/hdev/cli/enable/xrt
```

---

## Python Environment

```bash
cd hls-lut-accelerator
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

---

# 4. Repository Structure

```text
.
├── pwl_non_uniform.py        # Coefficient generation
├── pwl_non_uniform.cpp       # HLS kernel
├── pwl_non_uniform_tb.cpp    # Testbench
├── pwl_non_uniform.tcl       # HLS automation script
├── Makefile                  # Full Vitis build flow
└── lut_coeffs.h              # Generated header (not tracked)
```

---

# 5. Reproduction Instructions

## Quick Reproduction (Recommended Entry Point)

```bash
make clean
make build
```

This executes the full FPGA flow from kernel compilation to `.xclbin` generation.

---

## Step 1 — Coefficient Generation

```bash
cd hls-lut-accelerator
source .venv/bin/activate
python3 pwl_non_uniform.py
```

Output:
```text
lut_coeffs.h
```

---

## Step 2 — HLS Simulation and Synthesis

```bash
vitis_hls -f pwl_non_uniform.tcl
```

This performs:
- C simulation against golden model
- RTL synthesis (Verilog/VHDL generation)
- IP packaging (`.xo` / Vivado IP)

---

## Step 3 — FPGA Deployment

```bash
module load vivado/2024.1
source /opt/hdev/cli/enable/xrt
make build
```

Output:
```text
package.hw/kernels.xclbin
```

---

## Optional: Platform Selection

```bash
make PLATFORM=<platform_name> build
```

Supported examples:
- xilinx_u55c_gen3x16_xdma_3_202210_1
- xilinx_u280_gen3x16_xdma_1_202211_1
- xilinx_u250_gen3x16_xdma_4_1_202210_1

---

# 6. Evaluation Results

The following results were obtained using the full hardware implementation on AMD Alveo U55C.

## Numerical Accuracy (Fixed-Point)

| Function    | Mode        | Input Range | Format | RMSE     | Max Error | Status |
|-------------|------------|------------|--------|----------|-----------|--------|
| GELU        | Pointwise  | [-4, 4]    | <16,8> | 0.004626 | 0.012404  | PASS   |
| Exponential | Pointwise  | [-4, 4]    | <32,8> | 0.002157 | 0.004990  | PASS   |
| Softmax     | Multi-pass | [-4, 4]    | <32,8> | 0.000132 | 0.000597  | PASS   |

---

# 7. Reproducibility Notes

- Results depend on selected FPGA platform and Vitis version.
- Timing closure may vary across toolchain versions.
- Kernel frequency is fixed at 200 MHz unless modified in Makefile.
- A clean build is required for correct synthesis results.

If synthesis fails, run:
```bash
make cleanall
```

---

# 8. Expected Build Time

| Step              | Time Estimate |
|-------------------|--------------|
| HLS synthesis     | 5–20 min     |
| FPGA linking      | 30–120 min   |
| Full `v++` build  | 1–3 hours    |

---

# 9. Citation

```bibtex
@misc{pwl_nonuniform_fpga,
  title  = {Non-Uniform Piecewise Linear Approximation Accelerator for FPGA-based Machine Learning Inference},
  author = {Patrick Hugo Nepveu Nelson},
  year   = {2026},
  note   = {Engineering thesis (Tecnológico de Costa Rica) and conference submission under review. Hardware validated on AMD Alveo U55C.}
}
```

---

# 10. License

This project is licensed under the **Apache License 2.0**.

Copyright 2026 Patrick Hugo Nepveu Nelson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.