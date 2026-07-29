FROM runpod/pytorch:1.0.2-cu1281-torch260-ubuntu2204

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV CUDA_HOME=/usr/local/cuda
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.6+PTX"

RUN real_cuda=$(dirname $(dirname $(which nvcc 2>/dev/null) 2>/dev/null) 2>/dev/null); real_cuda=${real_cuda:-$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)}; if [ -n "$real_cuda" ] && [ "$real_cuda" != "/usr/local/cuda" ]; then ln -sfn "$real_cuda" /usr/local/cuda; fi; echo "CUDA_HOME target: $real_cuda"; ls -la /usr/local/cuda/bin/nvcc || true
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
    pip install --no-cache-dir pillow

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

RUN pip install --no-cache-dir runpod pillow rembg onnxruntime

COPY handler.py /app/handler.py
COPY api_server.py /app/api_server.py

# --- Self-contained DINOv3 (gated on HF; mirror via ModelScope) ---
RUN pip install --no-cache-dir modelscope && \
    python -c "from modelscope import snapshot_download; snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m', local_dir='/opt/dinov3')"
RUN sed -i "s|DINOv3ViTModel.from_pretrained(model_name)|DINOv3ViTModel.from_pretrained('/opt/dinov3')|" \
    /app/TRELLIS.2/trellis2/modules/image_feature_extractor.py && \
    grep -n "from_pretrained" /app/TRELLIS.2/trellis2/modules/image_feature_extractor.py | head -3


RUN sed -i "s|pipeline.rembg_model = getattr(rembg, args\['rembg_model'\]\['name'\])(\*\*args\['rembg_model'\]\['args'\])|pipeline.rembg_model = None|" \
    /app/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py && \
    grep -n "rembg_model" /app/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py | head -4
# Pre-download rembg u2net model so first generate() doesn't hang on download
RUN python -c "from rembg import new_session; new_session('u2net')" || echo "u2net predownload failed, will retry at runtime"

ENV PYTHONPATH=/app/TRELLIS.2
WORKDIR /app
CMD ["python", "-u", "api_server.py"]
