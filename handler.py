import runpod
import base64
import io
import os
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

PIPE = None

def get_pipe():
    global PIPE
    if PIPE is None:
        import torch
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        PIPE = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        PIPE.cuda()
    return PIPE


def handler(event):
    inp = event.get("input", {})
    # lightweight health/probe path
    if inp.get("ping"):
        try:
            import torch
            return {"ok": True, "cuda": torch.cuda.is_available(),
                    "loaded": PIPE is not None}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    try:
        img_b64 = inp.get("image_b64")
        resolution = inp.get("resolution")
        decimation = int(inp.get("decimation_target", 1_000_000))
        if not img_b64:
            return {"error": "image_b64 required"}

        from PIL import Image
        import o_voxel
        image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")

        pipe = get_pipe()
        kwargs = {"seed": int(inp.get("seed", 42))}
        if resolution:
            kwargs["resolution"] = int(resolution)
        mesh = pipe.run(image, **kwargs)[0]
        mesh.simplify(16_777_216)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation,
            texture_size=int(inp.get("texture_size", 4096)),
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        path = "/tmp/out.glb"
        glb.export(path, extension_webp=True)
        with open(path, "rb") as f:
            glb_b64 = base64.b64encode(f.read()).decode()
        return {"glb_b64": glb_b64, "engine": "trellis2-4b"}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-2000:]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
