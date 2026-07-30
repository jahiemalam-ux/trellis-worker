import base64, io, os, traceback, json, threading, uuid, subprocess, shutil, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

PIPE = None
LOAD_ERROR = None
BOOT_LOG = []
_REMBG_SESSION = None
JOBS = {}  # id -> {status, result, error}

BLENDER_BIN = os.environ.get("BLENDER_BIN", "/opt/blender/blender")
BLENDER_SCRIPT = os.environ.get("BLENDER_SCRIPT", "/app/blender_post.py")


def log(msg):
    BOOT_LOG.append(str(msg))
    print(msg, flush=True)

def eager_load():
    global PIPE, LOAD_ERROR
    try:
        import torch
        log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        PIPE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        try:
            PIPE.image_cond_model.cuda()
        except Exception:
            pass
        try:
            PIPE.to("cuda")
        except Exception:
            pass
        log("MODEL READY")
    except Exception as e:
        LOAD_ERROR = f"{e}\n{traceback.format_exc()[-2500:]}"
        log(f"LOAD FAILED: {e}")

def rembg_image(b64):
    from PIL import Image
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session
        _REMBG_SESSION = new_session("u2net")
    from rembg import remove as _rmbg
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return _rmbg(img, session=_REMBG_SESSION)

def run_multi(images, seed, pipeline_type, max_num_tokens=49152,
              ss_params=None, shape_params=None, tex_params=None, step=None):
    """Multi-view generation: average DINOv3 conditioning across views, then
    follow the exact same sampling flow as pipeline.run()."""
    import torch
    ss_params = ss_params or {}
    shape_params = shape_params or {}
    tex_params = tex_params or {}
    imgs = [PIPE.preprocess_image(im) for im in images]
    torch.manual_seed(seed)

    def fused_cond(res):
        conds = []
        for im in imgs:
            c = PIPE.get_cond([im], res, include_neg_cond=False)['cond']
            conds.append(c)
        c = torch.stack(conds).mean(dim=0)
        return {'cond': c, 'neg_cond': torch.zeros_like(c)}

    if step: step(f"fused cond over {len(imgs)} views")
    cond_512 = fused_cond(512)
    cond_1024 = fused_cond(1024) if pipeline_type != '512' else None
    ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
    coords = PIPE.sample_sparse_structure(cond_512, ss_res, 1, ss_params)
    if step: step("sparse structure done")
    if pipeline_type == '512':
        shape_slat = PIPE.sample_shape_slat(cond_512, PIPE.models['shape_slat_flow_model_512'], coords, shape_params)
        tex_slat = PIPE.sample_tex_slat(cond_512, PIPE.models['tex_slat_flow_model_512'], shape_slat, tex_params)
        res = 512
    elif pipeline_type == '1024':
        shape_slat = PIPE.sample_shape_slat(cond_1024, PIPE.models['shape_slat_flow_model_1024'], coords, shape_params)
        tex_slat = PIPE.sample_tex_slat(cond_1024, PIPE.models['tex_slat_flow_model_1024'], shape_slat, tex_params)
        res = 1024
    elif pipeline_type == '1024_cascade':
        shape_slat, res = PIPE.sample_shape_slat_cascade(
            cond_512, cond_1024,
            PIPE.models['shape_slat_flow_model_512'], PIPE.models['shape_slat_flow_model_1024'],
            512, 1024, coords, shape_params, max_num_tokens)
        tex_slat = PIPE.sample_tex_slat(cond_1024, PIPE.models['tex_slat_flow_model_1024'], shape_slat, tex_params)
    elif pipeline_type == '1536_cascade':
        shape_slat, res = PIPE.sample_shape_slat_cascade(
            cond_512, cond_1024,
            PIPE.models['shape_slat_flow_model_512'], PIPE.models['shape_slat_flow_model_1024'],
            512, 1536, coords, shape_params, max_num_tokens)
        tex_slat = PIPE.sample_tex_slat(cond_1024, PIPE.models['tex_slat_flow_model_1024'], shape_slat, tex_params)
    else:
        raise ValueError(f"Invalid pipeline type: {pipeline_type}")
    if step: step("shape+texture slat done")
    torch.cuda.empty_cache()
    return PIPE.decode_latent(shape_slat, tex_slat, res)

def export_glb(mesh, inp, out_path="/tmp/out.glb"):
    import o_voxel
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(inp.get("decimation_target", 1_000_000)),
        texture_size=int(inp.get("texture_size", 4096)),
        remesh=True, remesh_band=1, remesh_project=0, verbose=False)
    glb.export(out_path, extension_webp=True)
    return out_path


def b64_of(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def blender_finish(raw_glb, inp, step=None):
    """
    Run the headless Blender finishing pass: retopology, UV unwrap, texture bake.

    Returns (path, info). On any failure we return the raw mesh instead of
    failing the whole job — a dense mesh beats no mesh.
    """
    if not os.path.exists(BLENDER_BIN):
        raise RuntimeError(f"blender not found at {BLENDER_BIN}")

    out_path = os.path.join(tempfile.gettempdir(), f"clean_{uuid.uuid4().hex[:8]}.glb")
    cmd = [
        BLENDER_BIN, "--background", "--python", BLENDER_SCRIPT, "--",
        "--input", raw_glb,
        "--output", out_path,
        "--target-faces", str(int(inp.get("target_faces", 20000))),
        "--texture-size", str(int(inp.get("bake_texture_size", inp.get("texture_size", 2048)))),
        "--bake-samples", str(int(inp.get("bake_samples", 1))),
    ]
    if inp.get("no_bake"):
        cmd.append("--no-bake")

    if step: step("blender: starting")
    timeout = int(inp.get("blender_timeout", 900))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    # Blender is chatty; keep only our own tagged lines for the job log.
    tagged = [l for l in (proc.stdout or "").splitlines() if "[blender_post]" in l]
    tail = tagged[-14:] if tagged else (proc.stdout or "")[-1200:].splitlines()

    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            f"blender exit={proc.returncode}\n"
            + "\n".join(tail)
            + "\nSTDERR:\n" + (proc.stderr or "")[-800:]
        )

    if step: step("blender: done")
    return out_path, tail


def run_job(jid, inp):
    JOBS[jid]["status"] = "running"
    JOBS[jid]["steps"] = []
    def step(m):
        JOBS[jid]["steps"].append(m)
        print(f"[{jid}] {m}", flush=True)
    try:
        # --- Stage 0: text -> image (only when no image was supplied) ---
        b64s = inp.get("images_b64") or ([inp["image_b64"]] if inp.get("image_b64") else None)
        prompt_image_b64 = None
        if not b64s:
            prompt = inp.get("prompt")
            if not prompt:
                raise ValueError("provide image_b64, images_b64, or prompt")
            step("txt2img: loading")
            import txt2img
            prompt_image_b64 = txt2img.generate_b64(
                prompt,
                seed=int(inp.get("seed", 42)),
                steps=int(inp.get("t2i_steps", 4)),
                size=int(inp.get("t2i_size", 1024)),
                style_wrap=bool(inp.get("t2i_style_wrap", True)),
            )
            step("txt2img: image generated")
            if inp.get("t2i_unload", True):
                # Give the VRAM back before the 4B pipeline runs.
                txt2img.unload()
                step("txt2img: unloaded")
            b64s = [prompt_image_b64]

        step("decode+rembg start")
        images = []
        for b in b64s:
            images.append(rembg_image(b))
        step(f"rembg done x{len(images)}")

        seed = int(inp.get("seed", 42))
        pipeline_type = inp.get("pipeline_type", "1024_cascade")
        max_tokens = int(inp.get("max_num_tokens", 49152))
        step("pipeline start")
        if len(images) > 1:
            out = run_multi(images, seed, pipeline_type, max_tokens, step=step)
            mesh = out[0]
        else:
            mesh = PIPE.run(images[0], seed=seed, pipeline_type=pipeline_type,
                            max_num_tokens=max_tokens, preprocess_image=True)[0]
        step("pipeline done")
        mesh.simplify(16_777_216)

        raw_glb = export_glb(mesh, inp)
        step("glb exported")

        result = {"engine": "trellis2-4b"}
        if prompt_image_b64 and inp.get("return_prompt_image", True):
            result["prompt_image_b64"] = prompt_image_b64

        # --- Stage 4: Blender finishing (opt-in) ---
        if inp.get("blender_post", False):
            try:
                clean_glb, blog = blender_finish(raw_glb, inp, step=step)
                result["glb_b64"] = b64_of(clean_glb)
                result["raw_glb_b64"] = b64_of(raw_glb) if inp.get("return_raw", False) else None
                result["blender_log"] = blog
                result["postprocessed"] = True
            except Exception as e:
                step(f"blender failed, returning raw: {e}")
                result["glb_b64"] = b64_of(raw_glb)
                result["postprocessed"] = False
                result["blender_error"] = str(e)[:1500]
        else:
            result["glb_b64"] = b64_of(raw_glb)
            result["postprocessed"] = False

        result = {k: v for k, v in result.items() if v is not None}
        JOBS[jid]["result"] = result
        step("job complete")
        JOBS[jid]["status"] = "done"
    except Exception as e:
        JOBS[jid]["error"] = f"{e}\n{traceback.format_exc()[-2500:]}"
        JOBS[jid]["status"] = "failed"

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            blender_ok = os.path.exists(BLENDER_BIN)
            t2i = {"loaded": False}
            try:
                import txt2img
                t2i = txt2img.status()
            except Exception as e:
                t2i = {"loaded": False, "import_error": str(e)[:200]}
            self._send(200, {"ok": LOAD_ERROR is None, "loaded": PIPE is not None,
                             "load_error": LOAD_ERROR,
                             "blender": blender_ok, "blender_bin": BLENDER_BIN,
                             "txt2img": t2i,
                             "log": BOOT_LOG[-25:]})
        elif self.path.startswith("/job/"):
            jid = self.path[5:]
            j = JOBS.get(jid)
            if not j:
                return self._send(404, {"error": "unknown job"})
            out = {"status": j["status"], "steps": j.get("steps", [])}
            if j["status"] == "done":
                out["result"] = j["result"]
            if j["status"] == "failed":
                out["error"] = j["error"]
            self._send(200, out)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            inp = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})
        if self.path in ("/generate", "/text-to-3d"):
            if PIPE is None:
                return self._send(200, {"error": "model not loaded", "load_error": LOAD_ERROR})
            # /text-to-3d is /generate with the finishing stage on by default.
            if self.path == "/text-to-3d":
                inp.setdefault("blender_post", True)
            jid = uuid.uuid4().hex[:12]
            JOBS[jid] = {"status": "queued"}
            threading.Thread(target=run_job, args=(jid, inp), daemon=True).start()
            self._send(200, {"job_id": jid})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    log("api_server v5 starting (trellis2 + blender + txt2img), eager-loading model")
    log(f"blender present: {os.path.exists(BLENDER_BIN)} at {BLENDER_BIN}")
    threading.Thread(target=eager_load, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
