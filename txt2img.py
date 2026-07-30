"""
Text-to-image front-end, so the service accepts prompts instead of only images.

TRELLIS.2 is image-conditioned. Rodin takes text. This bridges the gap:
    prompt -> SDXL-Turbo image -> TRELLIS.2 -> mesh

VRAM discipline matters here. TRELLIS.2-4B already occupies a large slice of the
card, so this module:
  * lazy-loads (nothing is touched until a prompt actually arrives)
  * can evict itself with unload() so the 3D pipeline gets the memory back
  * uses SDXL-Turbo (~7GB fp16, 1-4 steps) instead of FLUX (~24GB) on purpose

Prompts get a light "clean single object" wrapper, because TRELLIS reconstructs best
from an isolated, centered, evenly-lit subject — the same reason rembg runs downstream.
"""

import gc
import io
import base64
import threading
import traceback

MODEL_ID = "stabilityai/sdxl-turbo"

_PIPE = None
_LOCK = threading.Lock()
_LOAD_ERROR = None

# TRELLIS wants an isolated subject on a plain background, not a photograph
# with scenery. Steering the image generator produces markedly better meshes.
POSITIVE_SUFFIX = (
    "single centered object, full object visible, plain neutral background, "
    "even diffuse studio lighting, sharp focus, product photograph, "
    "no shadows on background, 3/4 view"
)
NEGATIVE_PROMPT = (
    "multiple objects, cropped, cut off, partial view, busy background, scenery, "
    "text, watermark, people, hands, harsh shadows, motion blur, reflections, "
    "extreme close-up, tiling"
)


def log(msg):
    print(f"[txt2img] {msg}", flush=True)


def _build_prompt(prompt, style_wrap=True):
    p = prompt.strip()
    if style_wrap:
        p = f"{p}, {POSITIVE_SUFFIX}"
    return p


def load(device="cuda"):
    """Load SDXL-Turbo. Safe to call repeatedly; returns the cached pipe."""
    global _PIPE, _LOAD_ERROR
    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        try:
            import torch
            from diffusers import AutoPipelineForText2Image

            log(f"loading {MODEL_ID}")
            pipe = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            pipe.set_progress_bar_config(disable=True)
            try:
                pipe.to(device)
            except Exception as e:
                # Not enough free VRAM alongside TRELLIS — stream layers instead.
                log(f"direct .to({device}) failed ({e}); enabling cpu offload")
                pipe.enable_model_cpu_offload()
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

            _PIPE = pipe
            _LOAD_ERROR = None
            log("SDXL-TURBO READY")
            return _PIPE
        except Exception as e:
            _LOAD_ERROR = f"{e}\n{traceback.format_exc()[-2000:]}"
            log(f"LOAD FAILED: {e}")
            raise


def unload():
    """Free the image model so the 3D pipeline can reclaim VRAM."""
    global _PIPE
    with _LOCK:
        if _PIPE is None:
            return False
        try:
            import torch

            del _PIPE
            _PIPE = None
            gc.collect()
            torch.cuda.empty_cache()
            log("unloaded, VRAM released")
            return True
        except Exception as e:
            log(f"unload issue: {e}")
            _PIPE = None
            return False


def status():
    return {"loaded": _PIPE is not None, "model": MODEL_ID, "load_error": _LOAD_ERROR}


def generate(
    prompt,
    seed=42,
    steps=4,
    guidance=0.0,
    size=1024,
    style_wrap=True,
    negative_prompt=None,
):
    """
    Render a prompt to a PIL image suitable for TRELLIS conditioning.

    SDXL-Turbo is distilled: guidance must stay at 0.0 and steps at 1-4, or
    output degrades badly. Values are clamped rather than trusted.
    """
    import torch

    pipe = load()

    steps = max(1, min(int(steps), 8))
    # Turbo is trained for CFG-free sampling; a nonzero value here wrecks it.
    guidance = 0.0 if guidance is None else float(guidance)
    if guidance > 0.0:
        log(f"guidance {guidance} ignored (turbo is CFG-free)")
        guidance = 0.0

    size = max(512, min(int(size), 1024))
    full = _build_prompt(prompt, style_wrap)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    log(f"generating: steps={steps} size={size} seed={seed} :: {full[:90]}")
    kwargs = dict(
        prompt=full,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=size,
        width=size,
        generator=gen,
    )
    # Turbo at CFG 0 ignores negatives; only pass when it would actually apply.
    if negative_prompt or guidance > 0:
        kwargs["negative_prompt"] = negative_prompt or NEGATIVE_PROMPT

    image = pipe(**kwargs).images[0]
    log("image generated")
    return image


def generate_b64(prompt, **kw):
    """Same as generate(), but returns base64 PNG for the HTTP layer."""
    img = generate(prompt, **kw)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
