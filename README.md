# QUANTIZATION AND DPU DEPLOYMENT FOR CUSTOM - LAYER YOLO MODEL (YOLOvDP - ENHANCED) ON XILINX ZCU102

This repository guides you through quantizing, compiling, and deploying a custom-layer YOLOv5 model (`YOLOvDP-Enhanced`) from PyTorch FP32 (`best.pt`) to an INT8 `.xmodel` running on the **Xilinx ZCU102 DPU**.

---

## **💡 THE  STORY:**
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

To perform INT8 quantization, model inspection, and compilation, you must prepare the host environment with the official Vitis AI Docker container. You can follow the [Official Vitis AI MPSoC Setup Guide](https://xilinx.github.io/Vitis-AI/3.0/html/docs/quickstart/mpsoc.html) or run the commands below.

#### Step 1: Install Ubuntu via WSL2 (Windows PowerShell)

If operating on a Windows machine, set up an Ubuntu 20.04 environment using Windows Subsystem for Linux (WSL2):

```powershell
# Installs Ubuntu 20.04 distribution via WSL
wsl --install -d Ubuntu-20.04

# Lists all available online Linux distributions
wsl --list --online

# Launches the Ubuntu 20.04 environment
wsl -d Ubuntu-20.04
```

#### Step 2: Verify Docker on Ubuntu WSL
After launching Ubuntu WSL and installing Docker Desktop (with WSL 2 backend integration enabled), run the following commands to clone Vitis AI and verify your container runtime:

```host
# Clones the official Xilinx Vitis AI repository
git clone [https://github.com/Xilinx/Vitis-AI](https://github.com/Xilinx/Vitis-AI)

# Verifies Docker runtime permissions and connectivity
docker run hello-world

# Checks the installed Docker client and engine versions
docker --version
```

#### Step 3: Pull Vitis AI Docker Image
Pull the CPU-based PyTorch Docker image maintained by Xilinx:
```host
docker pull xilinx/vitis-ai-pytorch-cpu:latest
```

#### Step 4: Launch Container & Configure Environment
Navigate to the root directory of the cloned Vitis-AI repo, start the Docker container, and activate the required Python dependencies:

```host
# Launches the Vitis AI Docker container with workspace auto-mounted
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest

# Activates the target PyTorch conda environment configured for Vitis AI quantization
conda activate vitis-ai-pytorch

# Installs supplementary packages required by the customized YOLOv5 architecture
pip install seaborn timm efficientnet_pytorch
```

> [!NOTE]
> **Troubleshooting & Optimization Tips:**
> 
> * **WSL2 Memory Usage:** If you encounter performance issues with Ubuntu or WSL consuming excessive host RAM, you can limit the WSL virtual machine memory allocation (e.g., capping it at 4GB or 8GB depending on your quantization workload; setting `memory=1GB` for basic terminal operations). Check this setup guide: [WSL Memory Configuration Guide](https://youtu.be/urDkRPVvd88?si=wpjNyLsrLhyoyzaP).
> * **Docker Disk Space Management:** The Vitis AI Docker container images are quite large and can quickly consume disk capacity. If Docker is taking up too much storage on your system drive, refer to this walkthrough: [Managing & Moving Docker Disk Usage](https://www.youtube.com/watch?v=MFtdjhwC1co).

## 2. 🧮 Model Quantization Workflow (`best.pt` to `.xmodel`)
