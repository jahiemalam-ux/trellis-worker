FROM runpod/pytorch:1.0.2-cu1281-torch260-ubuntu2204

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1 libglib2.0-0 ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Clone with submodules (o-voxel lives in a submodule)
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2 /app/TRELLIS.2

WORKDIR /app/TRELLIS.2

# Official installer: base deps + all CUDA extensions (flash-attn, cumesh,
# o-voxel, flexgemm, nvdiffrast, nvdiffrec). --basic skips conda env creation.
RUN . ./setup.sh --basic --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec

RUN pip install --no-cache-dir runpod pillow rembg

COPY handler.py /app/handler.py

# Warm the model into the image so workers skip the 4B download on cold start
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B')" || true

WORKDIR /app
CMD ["python3", "-u", "handler.py"]
