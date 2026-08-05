# QUANTIZATION AND DPU DEPLOYMENT FOR CUSTOM-LAYER YOLO MODEL (YOLOVDP-ENHANCED) ON XILINX ZCU102

This repository guides you through quantizing, compiling, and deploying a custom-layer YOLOv5 model (`YOLOvDP-Enhanced`) from PyTorch FP32 (`best.pt`) to an INT8 `.xmodel` running on the **Xilinx ZCU102 DPU**.

---

**The Story:**
I trained a custom object detection model based on the YOLOv5 architecture, integrated with custom layer structures (**YOLOvDP-Enhanced**), and obtained the floating-point weight file (`best.pt`).

**The main challenge:**
How can I successfully deploy and run this custom model to perform real-time image detection on the Xilinx ZCU102 DPU board?

> 📌 **Note:** The pre-trained weights (`best.pt`) and model source code are available at:  
> 👉 [Enhanced-YOLOvDP-Fps-Upgrade](https://github.com/khangvohoang3007/Enhanced-YOLOvDP-Fps-Upgrade)

Because the Xilinx DPU hardware only supports a fixed set of standard neural network operators, deploying a model with custom layers requires resolving operator incompatibility, performing INT8 quantization via Vitis AI, and compiling the model into an executable `.xmodel`.

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
