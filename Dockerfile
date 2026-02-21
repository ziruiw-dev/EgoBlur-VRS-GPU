FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LIBTORCH_HOME=/opt/libtorch
ENV CMAKE_PREFIX_PATH=$LIBTORCH_HOME
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9"
ENV TORCHVISION_DIR=/usr/local/share/cmake/TorchVision
ENV DEBIAN_FRONTEND=noninteractive

# Install Core System Tools and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    unzip \
    ca-certificates \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python-is-python3 \
    libffi-dev \
    libpng-dev \
    libturbojpeg-dev \
    libopencv-dev \
    ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Heavy PyTorch and Python dependencies (Stability Layer)
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install "numpy<2" && \
    python3 -m pip install torch==2.1.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install ninja

# Install CMake 3.28+ (Static Binary Layer)
RUN mkdir -p /opt/cmake && cd /opt/cmake && \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.0-rc4/cmake-3.28.0-rc4-linux-x86_64.sh && \
    chmod +x cmake-3.28.0-rc4-linux-x86_64.sh && \
    ./cmake-3.28.0-rc4-linux-x86_64.sh --skip-license --prefix=/opt/cmake && \
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm cmake-3.28.0-rc4-linux-x86_64.sh

# Install LibTorch 2.1.0 (Static Binary Layer)
RUN mkdir -p /opt && cd /opt && \
    wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip -O libtorch.zip && \
    unzip libtorch.zip && rm libtorch.zip

# Build TorchVision from source (Stability Layer)
# This provides the necessary C++ headers and CMake files for find_package(TorchVision)
RUN mkdir -p /workspace/repos && cd /workspace/repos && \
    git clone --branch v0.16.0 https://github.com/pytorch/vision/ && \
    cd vision && \
    mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST \
      -DWITH_CUDA=ON \
      -DTorch_DIR=/opt/libtorch/share/cmake/Torch && \
    make -j$(nproc) && \
    make install

# Install App-Specific/Environment Dependencies (Most likely to change)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfmt-dev \
    liblz4-dev \
    libzstd-dev \
    libxxhash-dev \
    libboost-system-dev \
    libboost-iostreams-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-chrono-dev \
    libboost-date-time-dev \
    libomp-dev \
    libssl-dev \
    zlib1g-dev \
    libopus-dev \
    pkg-config \
    make \
    gcc \
    g++ \
    vim \
    tzdata \
    && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
    && echo "America/Los_Angeles" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/local/lib/python3.10/dist-packages/ \
    && chmod -R a+rw /usr/local/lib/python3.10/dist-packages/


# Get source code from github
RUN mkdir -p /workspace/repos && cd /workspace/repos && \
    git clone https://github.com/ziruiw-dev/EgoBlur-VRS-GPU.git

# Build ego_blur_vrs_mutation
RUN mkdir -p /workspace/repos/EgoBlur-VRS-GPU/gen1/tools/vrs_mutation/build && \
    cd /workspace/repos/EgoBlur-VRS-GPU/gen1/tools/vrs_mutation/build && \
    cmake .. -DTorch_DIR=$LIBTORCH_HOME/share/cmake/Torch -DTorchVision_DIR=$TORCHVISION_DIR && \
    make -j ego_blur_vrs_mutation

# Set workdir
WORKDIR /workspace

# Runscript equivalent: print info and execute command
ENTRYPOINT ["/bin/bash", "-c", "echo \"Welcome to EgoBlur Docker Container\"; echo \"LibTorch path: $LIBTORCH_HOME\"; exec \"$@\"", "--"]
CMD ["/bin/bash"]
