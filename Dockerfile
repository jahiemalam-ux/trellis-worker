FROM runpod/pytorch:1.0.2-cu1281-torch260-ubuntu2204

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1 libglib2.0-0 ninja-build libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/microsoft/TRELLIS.2 /app/TRELLIS.2
WORKDIR /app/TRELLIS.2

RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh transformers tensorboard pandas lpips zstandard kornia timm && \
    pip install --no-cache-dir \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 && \
    pip install --no-cache-dir pillow-simd

RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation || \
    (git clone --recursive https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
     cd /tmp/flash-attention && git checkout v2.7.3 && \
     pip install . --no-build-isolation && cd /app/TRELLIS.2)

RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast && \
    pip install /tmp/ext/nvdiffrast --no-build-isolation
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec && \
    pip install /tmp/ext/nvdiffrec --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh && \
    pip install /tmp/ext/CuMesh --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM && \
    pip install /tmp/ext/FlexGEMM --no-build-isolation
RUN pip install ./o-voxel --no-build-isolation

RUN pip install --no-cache-dir runpod pillow rembg

COPY handler.py /app/handler.py
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B')" || true

WORKDIR /app
CMD ["python3", "-u", "handler.py"]
