ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    HALF=1 \
    MAX_DET=100 \
    YOLO_IMGSZ=1536 \
    CONFIDENCE_THRESHOLD=0.20 \
    MIN_BOX_SIZE=10 \
    NMS_IOU_THRESHOLD=0.45 \
    POST_CONFIDENCE_THRESHOLD=0.0 \
    MAX_OUTPUT_PER_IMAGE=0 \
    MERGED_CLASS_POLICY=common \
    CLASS_MAPPING=/app/class_mapping.json \
    PIL_LOG_LEVEL=ERROR

RUN mkdir -p /app /saisresult

RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            /etc/apt/sources.list; \
    fi; \
    find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \) -exec sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        {} +; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        tini \
        bash \
        wget \
        ca-certificates \
        libcublas-12-0 \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6; \
    rm -rf /var/lib/apt/lists/*; \
    python3 -m pip install --upgrade "pip<25" setuptools wheel

RUN set -eux; \
    mkdir -p /usr/local/cuda/lib64; \
    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf; \
    for lib in libcublas libcublasLt libcudnn; do \
        target="$(find -H /usr/local/cuda /usr/local/cuda-* /usr/lib -name "${lib}.so.*" 2>/dev/null | sort -V | tail -n 1 || true)"; \
        if [ -n "$target" ]; then \
            ln -sf "$target" "/usr/local/cuda/lib64/${lib}.so"; \
            echo "Linked /usr/local/cuda/lib64/${lib}.so -> $target"; \
        else \
            echo "Missing ${lib}.so.*"; \
        fi; \
    done; \
    ldconfig; \
    python3 -c "import ctypes; [ctypes.CDLL(x) for x in ('libcublas.so', 'libcublasLt.so', 'libcudnn.so')]; print('CUDA libraries load OK')"

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Install PyTorch with CUDA 12.4 support (matches eval environment)
RUN python3 -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --prefer-binary -r /app/requirements.txt

# Verify CUDA + PyTorch
RUN python3 -c "import torch; print(f'torch={torch.__version__}, CUDA available={torch.cuda.is_available()}, Devices={torch.cuda.device_count()}')"

# Inference code
COPY src/ /app/src/

# Model weights for inference
COPY classifier_output/best.pth /app/classifier_output/best.pth
COPY yolo_dataset/detect_yolo11l/weights/best.pt /app/yolo_dataset/weights/best.pt

# Character mapping: folder ID → Chinese character
COPY class_mapping.json /app/class_mapping.json

# Entry point
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/bin/bash", "/app/run.sh"]
