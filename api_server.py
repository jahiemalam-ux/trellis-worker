import base64, io, os, traceback, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

PIPE = None
LOAD_ERROR = None
BOOT_LOG = []

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
        PIPE.cuda()
        log("MODEL READY")
    except Exception as e:
        LOAD_ERROR = f"{e}\n{traceback.format_exc()[-2500:]}"
        log(f"LOAD FAILED: {e}")

def generate(inp):
    img_b64 = inp.get("image_b64")
    if not img_b64:
        return {"error": "image_b64 required"}
    from PIL import Image
    import o_voxel
    image = Image.open(io.BytesIO(base64.b64decode(img_b64)))
    # Convert to RGBA with opaque alpha -> pipeline skips gated RMBG-2.0.
    # (Golf photos are already background-removed upstream.)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    pipe = PIPE
    kwargs = {"seed": int(inp.get("seed", 42))}
    if inp.get("resolution"):
        kwargs["resolution"] = int(inp["resolution"])
    mesh = pipe.run(image, **kwargs)[0]
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
                return self._send(200, {"error": "model not loaded yet", "load_error": LOAD_ERROR,
                                        "log": BOOT_LOG[-15:]})
            try:
                self._send(200, generate(inp))
            except Exception as e:
                self._send(200, {"error": str(e), "trace": traceback.format_exc()[-3000:]})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    log("api_server starting, eager-loading model")
    t = threading.Thread(target=eager_load, daemon=True)
    t.start()
    ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
