# EgoBlur VRS GPU

This repository is a modified extension of the [EgoBlur](https://github.com/facebookresearch/EgoBlur) repository (specifically based on commit `e5ed753`). 

It provides a **Dockerized environment** to easily build and run the `ego_blur_vrs_mutation` tool, which is used for blurring faces and license plates directly in Aria Gen1 VRS files using GPU acceleration.

## Key Changes
As required by the Apache License 2.0, the following changes and additions have been made to the original EgoBlur repository:
- `Dockerfile`: Added a multi-layer build environment with CUDA 12.1, LibTorch, and TorchVision.
- `docker-compose.yml`: Added a service definition for the EgoBlur container.
- `blur_vrs.py`: Added a Python wrapper to handle dynamic container execution and volume mounting.
- `blur_vrs_gpu.sh`: Added a shell script wrapper for the Python script.
- `gen1/tools/vrs_mutation/CMakeLists.txt`: Modified to support CUDA detection and linking with TorchVision properly within the Docker environment.

## Prerequisites
- **Linux** (Tested on Ubuntu 22.04)
- **NVIDIA GPU** (Tested on RTX 4090)
- **Docker** and **Docker Compose**
- **NVIDIA Container Toolkit** installed and configured

## Setup

### 1. Model Checkpoints
Download the EgoBlur JIT model checkpoints from: [https://www.projectaria.com/tools/egoblur/](https://www.projectaria.com/tools/egoblur/)

You need:
- `ego_blur_face.jit`
- `ego_blur_lp.jit`

save them to `/some_dir/to/model/checkpoints/`

### 2. Obtain the Docker Image

You can either pull the pre-built image from a registry or build it yourself if you wish to modify the source code.

#### Option A: Pull the image
If the image is hosted on a registry (e.g., Docker Hub or GitHub Packages):
```bash
docker pull ziruiw/egoblur-vrs-gpu:1.0
```
#### Option B: Build it yourself
If you want to bake your local changes into the image:
```bash
docker compose build
```

## Usage

You can run the blurring tool directly using the Python wrapper

```bash
python blur_vrs.py \
    --input-dir /dir/to/vrs/raw \
    --output-dir /dir/to/vrs/blur \
    --vrs-name recording.vrs \
    --ckpt-dir /some_dir/to/model/checkpoints/ \
    --face-conf 0.85 \
    --lp-conf 0.85
```

## Project Structure
- `Dockerfile`: Multi-layer build environment with CUDA 12.1, LibTorch, and TorchVision.
- `docker-compose.yml`: Service definition for the EgoBlur container.
- `blur_vrs.py`: Python wrapper for dynamic container execution.
- `blur_vrs_gpu.sh`: Shell script wrapper for the Python script.
- `gen1/`: Original EgoBlur Gen1 source code and utilities.

## Acknowledgements
This project is based on [EgoBlur](https://github.com/facebookresearch/EgoBlur) by Meta. 


The Dockerfile configuration was inspired by the discussions in [EgoBlur Issue #12](https://github.com/facebookresearch/EgoBlur/issues/12) and the Apptainer implementation provided by [@ashwanirathee](https://github.com/ashwanirathee) in this [gist](https://gist.github.com/ashwanirathee/caf2534626f05dcc9f9acf15d14b7ece).
