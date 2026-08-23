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
RUN pip install --no-cache-dir --upgrade 'transformers>=4.56.0' && pip install --no-cache-dir modelscope && \
    python -c "from modelscope import snapshot_download; snapshot_download('facebook/dinov3-vitl16-pretrain-lvd1689m', local_dir='/opt/dinov3')"
# Patch DinoV3FeatureExtractor to load from local mirror (code is correct for transformers>=4.56)
RUN python - <<'PYFIX'
p = "/app/TRELLIS.2/trellis2/modules/image_feature_extractor.py"
s = open(p).read()
s = s.replace("DINOv3ViTModel.from_pretrained(model_name)", "DINOv3ViTModel.from_pretrained('/opt/dinov3')")
open(p, "w").write(s)
print("patched path:", "/opt/dinov3" in s)
PYFIX


RUN sed -i "s|pipeline.rembg_model = getattr(rembg, args\['rembg_model'\]\['name'\])(\*\*args\['rembg_model'\]\['args'\])|pipeline.rembg_model = None|" \
    /app/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py && \
    grep -n "rembg_model" /app/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py | head -4
# Pre-download rembg u2net model so first generate() doesn't hang on download
RUN python -c "from rembg import new_session; new_session('u2net')" || echo "u2net predownload failed, will retry at runtime"

# Hard-pin transformers>=4.56 (DINOv3ViTModel.layer) AFTER all installs so nothing downgrades it
RUN pip install --no-cache-dir --force-reinstall --no-deps 'transformers==4.56.0' && \
    pip install --no-cache-dir 'tokenizers>=0.20' 'safetensors' 'huggingface-hub==0.34.4' 'regex' 'pyyaml' 'numpy' 'packaging' 'tqdm' 'requests' && \
    python -c "import transformers; from transformers import DINOv3ViTModel; print('transformers', transformers.__version__)"

# --- Blender (headless) for retopology / UV / texture bake ---
# Tarball rather than pip `bpy`: the wheel pins an exact Python version and the
# base image's interpreter is not guaranteed to match.
ARG BLENDER_VERSION=5.2.0
ARG BLENDER_SERIES=5.2
RUN apt-get update && apt-get install -y --no-install-recommends \
    xz-utils libxi6 libxxf86vm1 libxfixes3 libxrender1 libsm6 libice6 \
    libxkbcommon0 libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
      -o /tmp/blender.tar.xz && \
    mkdir -p /opt/blender && \
    tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 && \
    rm /tmp/blender.tar.xz && \
    /opt/blender/blender --background --version
ENV BLENDER_BIN=/opt/blender/blender
ENV BLENDER_SCRIPT=/app/blender_post.py

# --- Text-to-image front-end (SDXL-Turbo) so prompts work, not just images ---
# --no-deps guards the pinned transformers==4.56.0 that DINOv3 requires.
RUN pip install --no-cache-dir --no-deps diffusers==0.31.0 && \
    pip install --no-cache-dir accelerate sentencepiece && \
    python -c "import diffusers; print('diffusers', diffusers.__version__)" && \
    python -c "import transformers; print('transformers', transformers.__version__)"

# SDXL-Turbo weights are deliberately NOT baked in: ~7GB would blow the CI runner's
# ~14GB disk and slow every pod cold start. txt2img.py lazy-loads on first prompt.
# Set HF_HOME to a persistent volume if you want to cache them across pods.

COPY blender_post.py /app/blender_post.py
COPY txt2img.py /app/txt2img.py
COPY subject_check.py /app/subject_check.py

ENV PYTHONPATH=/app/TRELLIS.2:/app
WORKDIR /app
CMD ["python", "-u", "api_server.py"]
