# Quantization & DPU Deployment for YOLOvDP-Enhanced on Xilinx ZCU102 🚀

This repository provides a step-by-step guide on how to quantize, compile, and deploy **YOLOvDP-Enhanced**—a customized variant based on the **YOLOv5** architecture featuring custom layers—from PyTorch (`best.pt`) to an INT8 compiled `.xmodel` running on the Deep Learning Processing Unit (DPU) of the **Xilinx Zynq UltraScale+ MPSoC ZCU102** evaluation board.

---

## 1. ❓ Problem Statement

After training a custom **YOLOvDP-Enhanced** model (modified from YOLOv5 with custom layer structures) and obtaining the floating-point weight file (`best.pt`), the challenge is:

> **How can we bypass unsupported operator constraints on the Xilinx DPU, perform INT8 quantization via Vitis AI, and compile the PyTorch model into an executable `.xmodel` to perform real-time image detection directly on the ZCU102 hardware?**

---

## 2. 🛠️ Prerequisites & System Setup

### 2.1 Model & Custom Layer Requirements
Because the Xilinx DPU only supports a fixed set of hardware operations:
- Custom layers must be mapped or re-implemented using standard PyTorch operators supported by the Vitis AI PyTorch Quantizer (`pytorch_nndct`).
- Activation functions (e.g., `SiLU`, `Mish`) should be evaluated for DPU compatibility or mapped to hardware-friendly alternatives like `ReLU` or `LeakyReLU` during quantization.
- The network should be structured to isolate the DPU-supported Backbone/Neck from any unsupported custom Post-processing/Detect heads (which can run fallback execution on the ZCU102 ARM CPU).

### 2.2 Host Environment Setup (Ubuntu, Docker, Vitis AI)
Perform these steps on your Linux Host PC/Server:

1. **Install Ubuntu 20.04/22.04 LTS** with Docker Engine and NVIDIA Container Toolkit (if using GPU acceleration).
2. **Pull the official Vitis AI Docker Image**:
   ```bash
   # CPU-only version
   docker pull xilinx/vitis-ai-pytorch-cpu:latest

   # GPU-accelerated version (Recommended for faster calibration)
   docker pull xilinx/vitis-ai-pytorch-gpu:latest
