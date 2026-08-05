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

## 1. 🛠️ Prerequisites & System Setup

### 1.1 Model & Custom Layer Requirements

Different Xilinx edge devices and DPU architectures support different sets of activation functions. For instance, platforms like the KV260 or ZCU102 may not natively support non-linear activations like **SiLU** on the DPU hardware, causing fallback to CPU or quantization errors. In our **YOLOvDP-Enhanced** model, all unsupported activations were explicitly replaced with **LeakyReLU** to ensure seamless INT8 quantization and full DPU hardware acceleration.

> 📌 **Useful Resources:**
> - **Supported Operators Reference:** Check the official [Vitis AI Supported Operators Documentation](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Currently-Supported-Operators) to verify operator compatibility for your target DPU architecture.
> - **Retrained Kaggle Notebook:** View our retrained model and weight preparation process on [Kaggle](https://www.kaggle.com/code/gautapcode/vitis-ai).

#### Required Code Modifications
When modifying a Ultralytics YOLOv5 repository to adapt custom layers and activation functions, you need to adjust several core files depending on your specific network design:

- `models/common.py` (Defines custom block structures and activation layer mappings)
- `models/yolo.py` (Handles parsing and building network architecture layers)
- `models/experimental.py` (Contains experimental modules and layer wrappers)
- `train.py` (Configures training parameters and model initialization)

> 💡 **Reference Guide:** For a detailed breakdown of replacing activation layers and modifying YOLOv5 source code for Vitis AI compatibility, refer to this [Hackster.io Guide on YOLOv5 Quantization with Vitis AI 3.0](https://www.hackster.io/LogicTronix/yolov5-quantization-compilation-with-vitis-ai-3-0-for-kria-7b005d).

### 1.2 Host Environment Setup (Ubuntu, Docker, Vitis AI)
Perform these steps on your Linux Host PC/Server:

1. **Install Ubuntu 20.04/22.04 LTS** with Docker Engine and NVIDIA Container Toolkit (if using GPU acceleration).
2. **Pull the official Vitis AI Docker Image**:
   ```bash
   # CPU-only version
   docker pull xilinx/vitis-ai-pytorch-cpu:latest

   # GPU-accelerated version (Recommended for faster calibration)
   docker pull xilinx/vitis-ai-pytorch-gpu:latest
