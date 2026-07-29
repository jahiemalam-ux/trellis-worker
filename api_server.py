import base64, io, os, traceback, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

PIPE = None
BOOT_LOG = []

def log(msg):
    BOOT_LOG.append(str(msg))
    print(msg, flush=True)

def get_pipe():
    global PIPE
    if PIPE is None:
        import torch
        log(f"torch {torch.__version__} cuda_avail={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"gpu={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        PIPE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        PIPE.cuda()
        log("pipeline loaded")
    return PIPE

def generate(inp):
    img_b64 = inp.get("image_b64")
    if not img_b64:
        return {"error": "image_b64 required"}
    from PIL import Image
    import o_voxel
    image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    pipe = get_pipe()
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
            try:
                import torch
                self._send(200, {"ok": True, "torch": torch.__version__,
                                 "cuda": torch.cuda.is_available(),
                                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                 "loaded": PIPE is not None, "log": BOOT_LOG[-20:]})
            except Exception as e:
                self._send(200, {"ok": False, "error": str(e), "trace": traceback.format_exc()[-2000:], "log": BOOT_LOG[-20:]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            inp = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})
        if self.path == "/generate":
            try:
                self._send(200, generate(inp))
            except Exception as e:
                self._send(200, {"error": str(e), "trace": traceback.format_exc()[-3000:], "log": BOOT_LOG[-20:]})
        elif self.path == "/preload":
            try:
                get_pipe()
                self._send(200, {"ok": True, "log": BOOT_LOG[-20:]})
            except Exception as e:
                self._send(200, {"ok": False, "error": str(e), "trace": traceback.format_exc()[-3000:]})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    log("api_server starting on :8080")
    ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
