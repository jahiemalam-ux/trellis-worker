import base64, io, os, traceback, json, threading, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

PIPE = None
LOAD_ERROR = None
BOOT_LOG = []
_REMBG_SESSION = None
JOBS = {}  # id -> {status, result, error}

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
        # Official app pattern: low_vram pipeline manages device moves per-call.
        # Just ensure the conditioning model is on GPU and stays there.
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

def generate(inp):
    from PIL import Image
    import o_voxel
    image = Image.open(io.BytesIO(base64.b64decode(inp["image_b64"]))).convert("RGB")
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        from rembg import new_session
        _REMBG_SESSION = new_session("u2net")
    from rembg import remove as _rmbg
    image = _rmbg(image, session=_REMBG_SESSION)
    kwargs = {"seed": int(inp.get("seed", 42))}
    if inp.get("pipeline_type"):
        kwargs["pipeline_type"] = inp["pipeline_type"]
    mesh = PIPE.run(image, **kwargs)[0]
    mesh.simplify(16_777_216)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(inp.get("decimation_target", 1_000_000)),
        texture_size=int(inp.get("texture_size", 4096)),
        remesh=True, remesh_band=1, remesh_project=0, verbose=False)
    glb.export("/tmp/out.glb", extension_webp=True)
    with open("/tmp/out.glb", "rb") as f:
        return {"glb_b64": base64.b64encode(f.read()).decode(), "engine": "trellis2-4b"}

def run_job(jid, inp):
    JOBS[jid]["status"] = "running"
    try:
        JOBS[jid]["result"] = generate(inp)
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
            self._send(200, {"ok": LOAD_ERROR is None, "loaded": PIPE is not None,
                             "load_error": LOAD_ERROR, "log": BOOT_LOG[-25:]})
        elif self.path.startswith("/job/"):
            jid = self.path[5:]
            j = JOBS.get(jid)
            if not j:
                return self._send(404, {"error": "unknown job"})
            out = {"status": j["status"]}
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
        if self.path == "/generate":
            if PIPE is None:
                return self._send(200, {"error": "model not loaded", "load_error": LOAD_ERROR})
            jid = uuid.uuid4().hex[:12]
            JOBS[jid] = {"status": "queued"}
            threading.Thread(target=run_job, args=(jid, inp), daemon=True).start()
            self._send(200, {"job_id": jid})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    log("api_server v3 starting, eager-loading model")
    threading.Thread(target=eager_load, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
