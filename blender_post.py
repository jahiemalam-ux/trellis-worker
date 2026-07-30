"""
Headless Blender post-processing for TRELLIS.2 output — the "Rodin finishing" stage.

Takes the dense, non-watertight, auto-UV'd mesh TRELLIS produces and turns it into
a clean game-ready asset: quad topology, sane UV islands, textures baked across.

Run:
  blender --background --python blender_post.py -- \
      --input /tmp/out.glb --output /tmp/clean.glb \
      --target-faces 20000 --texture-size 2048

Design notes:
  * Retopo is a fallback chain. TRELLIS meshes are frequently non-manifold, and
    QuadriFlow refuses non-manifold input, so we degrade gracefully rather than die.
  * Bake is DIFFUSE/COLOR only (selected_to_active). No lighting passes, so 1 sample
    is enough and it stays fast.
  * Everything is wrapped so a failure in the pretty path still yields a usable mesh.
"""

import bpy
import bmesh
import numpy as np
import sys
import os
import argparse
import traceback


def log(msg):
    print(f"[blender_post] {msg}", flush=True)


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target-faces", type=int, default=20000)
    p.add_argument("--texture-size", type=int, default=2048)
    p.add_argument("--bake-samples", type=int, default=1)
    p.add_argument("--cage-ratio", type=float, default=0.05,
                   help="bake cage extrusion as a fraction of the bounding box")
    p.add_argument("--no-bake", action="store_true")
    p.add_argument("--no-pbr", action="store_true",
                   help="skip normal/roughness map baking (baseColor only)")
    p.add_argument("--no-repair", action="store_true",
                   help="skip manifold repair (quads/watertight become unlikely)")
    p.add_argument("--no-decimate-fallback", action="store_true",
                   help="never fall back to decimation; keep topology closed even if over budget")
    p.add_argument("--repair-detail", type=int, default=200,
                   help="voxel divisor for the repair pass; higher preserves more detail")
    p.add_argument("--smooth-shading", action="store_true", default=True)
    return p.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def enable_gpu_cycles():
    """Point Cycles at CUDA if the box has it; silently fall back to CPU."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        found = False
        for d in prefs.devices:
            if d.type == "CUDA":
                d.use = True
                found = True
            else:
                d.use = False
        if found:
            scene.cycles.device = "GPU"
            log("cycles: GPU (CUDA)")
            return
    except Exception as e:
        log(f"cycles GPU setup failed ({e}), using CPU")
    scene.cycles.device = "CPU"
    log("cycles: CPU")


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh found in input GLB")
    # Join everything into one object so bake/remesh has a single source.
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    src = bpy.context.view_layer.objects.active
    src.name = "SOURCE"
    log(f"imported: {len(src.data.vertices)} verts / {len(src.data.polygons)} faces")
    return src


def mesh_stats(obj):
    return len(obj.data.vertices), len(obj.data.polygons)


def clean_mesh(obj):
    """Weld duplicate verts and drop loose geometry — helps QuadriFlow's odds."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    log(f"cleaned: {mesh_stats(obj)[0]} verts / {mesh_stats(obj)[1]} faces")


def is_manifold(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bad = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return bad == 0, bad


def count_holes(obj):
    """Boundary edges — a closed surface has none."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    n = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()
    return n


def fill_holes(obj, max_sides=0):
    """Cheap topological hole fill. max_sides=0 means no limit."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.mesh.fill_holes(sides=max_sides)
    except Exception as e:
        log(f"fill_holes failed: {e}")
    bpy.ops.object.mode_set(mode="OBJECT")


def make_manifold(obj, detail_divisor=200):
    """
    Force the mesh closed and manifold so QuadriFlow will accept it.

    This is the fix for both known defects: QuadriFlow is the only path that
    yields true quads AND a watertight result, but it refuses non-manifold
    input, so previously we fell through to decimation — which produces
    triangles and reopens holes. Repairing first means QuadriFlow actually runs.

    Escalates only as far as needed:
      1. weld + fill holes           (cheap, shape-preserving)
      2. OpenVDB voxel remesh        (guarantees a closed manifold shell)
    """
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    manifold, bad = is_manifold(obj)
    holes = count_holes(obj)
    log(f"repair: start manifold={manifold} bad_edges={bad} boundary_edges={holes}")

    if not manifold or holes:
        fill_holes(obj)
        manifold, bad = is_manifold(obj)
        holes = count_holes(obj)
        log(f"repair: after fill_holes manifold={manifold} bad_edges={bad} boundary_edges={holes}")

    if manifold and holes == 0:
        return True, "fill_holes"

    # Voxel remesh rebuilds the surface from a signed distance field, which is
    # closed and manifold by construction. Detail is set high here on purpose:
    # this pass exists to fix topology, and QuadriFlow reduces the count after.
    try:
        dims = obj.dimensions
        largest = max(dims.x, dims.y, dims.z) or 1.0
        obj.data.remesh_voxel_size = max(largest / detail_divisor, 1e-4)
        obj.data.remesh_voxel_adaptivity = 0.0
        bpy.ops.object.voxel_remesh()
        manifold, bad = is_manifold(obj)
        holes = count_holes(obj)
        log(f"repair: voxel_remesh @ {obj.data.remesh_voxel_size:.5f} -> "
            f"{mesh_stats(obj)[1]} faces manifold={manifold} boundary_edges={holes}")
        if holes:
            fill_holes(obj)
            holes = count_holes(obj)
        return (is_manifold(obj)[0] and holes == 0), "voxel_remesh"
    except Exception as e:
        log(f"repair: voxel_remesh failed: {e}")
        return False, "failed"


def retopologize(obj, target_faces, allow_decimate=True):
    """
    Reduce to the face budget, preferring topology that stays closed and quad.

      1. QuadriFlow at target_faces  -> true quads, watertight, exact budget
      2. Voxel remesh sized to hit the budget directly (NO trailing decimate,
         since decimating is what reopened holes before)
      3. Decimate  -> triangles, breaks watertight; only if allow_decimate

    Note there is deliberately no decimate after remeshing anymore.
    """
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    manifold, bad = is_manifold(obj)

    if manifold:
        try:
            # mode="FACES" is essential: the operator defaults to RATIO with
            # target_ratio=1.0, which rebuilds at the SAME density and silently
            # ignores target_faces (observed: 201870 in -> 201870 out).
            bpy.ops.object.quadriflow_remesh(
                mode="FACES",
                target_faces=target_faces,
                use_preserve_sharp=False,
                use_preserve_boundary=False,
                use_mesh_symmetry=False,
                smooth_normals=True,
            )
            achieved = mesh_stats(obj)[1]
            if achieved > target_faces * 3:
                # Guard against the operator no-op'ing again on a future version.
                log(f"WARNING: quadriflow returned {achieved} faces for a "
                    f"{target_faces} budget — target may have been ignored")
            log(f"retopo: quadriflow -> {mesh_stats(obj)[1]} faces "
                f"(watertight={count_holes(obj) == 0})")
            return "quadriflow"
        except Exception as e:
            log(f"quadriflow failed even on manifold input: {e}")
    else:
        log(f"retopo: skipping quadriflow, still non-manifold ({bad} edges)")

    # Search a voxel size that lands near the budget, instead of remeshing
    # coarsely and then decimating (which is what broke watertightness).
    try:
        dims = obj.dimensions
        largest = max(dims.x, dims.y, dims.z) or 1.0
        base = obj.data.copy()
        best = None
        lo, hi = largest / 400.0, largest / 12.0
        for _ in range(7):
            mid = (lo + hi) / 2.0
            obj.data = base.copy()
            obj.data.remesh_voxel_size = max(mid, 1e-4)
            obj.data.remesh_voxel_adaptivity = 0.0
            bpy.ops.object.voxel_remesh()
            n = mesh_stats(obj)[1]
            if best is None or abs(n - target_faces) < abs(best[1] - target_faces):
                best = (mid, n, obj.data.copy())
            if n > target_faces:
                lo = mid          # too dense -> larger voxels
            else:
                hi = mid
        if best:
            obj.data = best[2]
            log(f"retopo: voxel_remesh tuned @ {best[0]:.5f} -> {mesh_stats(obj)[1]} faces "
                f"(watertight={count_holes(obj) == 0}, no decimate)")
            return "voxel_remesh_tuned"
    except Exception as e:
        log(f"tuned voxel remesh failed: {e}")

    cur = mesh_stats(obj)[1]
    if allow_decimate and cur > target_faces:
        m = obj.modifiers.new("dec", "DECIMATE")
        m.decimate_type = "COLLAPSE"
        m.ratio = target_faces / cur
        bpy.ops.object.modifier_apply(modifier=m.name)
        log(f"retopo: decimate fallback -> {mesh_stats(obj)[1]} faces "
            f"(WARNING: triangles, may not be watertight)")
        return "decimate"

    log(f"retopo: left as-is at {cur} faces")
    return "none"


def unwrap(obj, angle_limit=66.0, island_margin=0.002):
    bpy.context.view_layer.objects.active = obj
    # Drop TRELLIS's generated UVs; we want islands that match the new topology.
    while obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[0])
    obj.data.uv_layers.new(name="UVMap")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.smart_project(
            angle_limit=angle_limit * (3.14159265 / 180.0),
            island_margin=island_margin,
            correct_aspect=True,
            scale_to_bounds=False,
        )
        method = "smart_project"
    except TypeError:
        # Older/newer Blender takes degrees instead of radians.
        bpy.ops.uv.smart_project(angle_limit=angle_limit, island_margin=island_margin)
        method = "smart_project"
    except Exception as e:
        log(f"smart_project failed: {e}, falling back to sphere unwrap")
        bpy.ops.uv.sphere_project()
        method = "sphere_project"
    bpy.ops.object.mode_set(mode="OBJECT")
    log(f"unwrap: {method}")
    return method


def make_bake_target(obj, texture_size):
    """Fresh material with an empty image node — Cycles bakes into this image."""
    img = bpy.data.images.new("BakedBaseColor", texture_size, texture_size, alpha=True)
    img.generated_color = (0, 0, 0, 1)

    mat = bpy.data.materials.new("BakedMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.name = "BAKE_TARGET"

    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    nt.nodes.active = tex
    return img, tex


def image_luminance(img):
    """Mean 0-255 luminance of a Blender image, for verifying a bake worked."""
    n = len(img.pixels)
    buf = np.empty(n, dtype=np.float32)
    img.pixels.foreach_get(buf)
    rgb = buf.reshape(-1, 4)[:, :3]
    return float(rgb.mean() * 255.0), float((rgb.max(axis=1) < 0.1).mean())


def _do_bake(src, tgt, extrusion, samples, margin):
    scene = bpy.context.scene
    scene.cycles.samples = max(samples, 1)
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.use_cage = False
    scene.render.bake.cage_extrusion = extrusion
    scene.render.bake.margin = margin
    scene.render.bake.use_clear = True

    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt  # active = bake destination

    bpy.ops.object.bake(
        type="DIFFUSE",
        pass_filter={"COLOR"},
        use_selected_to_active=True,
        cage_extrusion=extrusion,
        margin=margin,
    )


def bake_color(src, tgt, texture_size, samples, img, cage_ratio=0.05):
    """
    Bake SOURCE's base color onto TGT's new UVs via selected_to_active.

    cage_extrusion MUST scale with the object. It used to be hardcoded at 0.02,
    which happened to work on a small mug but failed badly on a car: the manifold
    repair displaces the retopo surface away from the original, and when that gap
    exceeds the extrusion the bake rays miss entirely and return black (observed:
    47% near-black texels, mean luminance 34/255 on a silver car).

    So: size the cage from the bounding box, then verify the result and escalate
    if it still came out dark.
    """
    dims = src.dimensions
    largest = max(dims.x, dims.y, dims.z) or 1.0
    margin = max(texture_size // 256, 4)

    attempts = [cage_ratio, cage_ratio * 3.0, cage_ratio * 8.0]
    best = None
    for i, ratio in enumerate(attempts):
        extrusion = max(largest * ratio, 1e-4)
        _do_bake(src, tgt, extrusion, samples, margin)
        lum, dark_frac = image_luminance(img)
        log(f"bake: attempt {i+1} extrusion={extrusion:.4f} "
            f"(={ratio:.3f} x {largest:.3f}) -> luminance={lum:.1f}/255 "
            f"dark_frac={dark_frac:.2f}")
        if best is None or lum > best[0]:
            best = (lum, extrusion, dark_frac)
        # A good bake on a lit subject lands well above this; near-black means
        # the rays are missing the source surface.
        if lum >= 45.0 and dark_frac <= 0.35:
            log(f"bake: OK on attempt {i+1}")
            return lum, dark_frac
        if i < len(attempts) - 1:
            log("bake: too dark, retrying with a larger cage")

    log(f"bake: WARNING still dark after {len(attempts)} attempts "
        f"(best luminance {best[0]:.1f}) — texture may be unusable")
    return best[0], best[2]


def bake_normal_map(src, tgt, texture_size, cage_ratio=0.05):
    """
    Bake a tangent-space normal map from the dense source onto the retopo mesh.

    This is the biggest single quality lever the audit surfaced: professional
    GLBs (DamagedHelmet: 15k faces, lum 116) carry a normal map that preserves
    all the fine surface relief we decimate away. Without it, a low-poly mesh
    reads as a smooth lump no matter how good the color is.
    """
    img = bpy.data.images.new(
        "BakedNormal", texture_size, texture_size, alpha=False,
        float_buffer=True,   # normals need >8-bit precision to avoid banding
    )
    img.colorspace_settings.name = "Non-Color"

    mat = tgt.data.materials[0]
    nt = mat.node_tree
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    node.name = "NORMAL_TARGET"
    nt.nodes.active = node

    dims = src.dimensions
    largest = max(dims.x, dims.y, dims.z) or 1.0
    ext = max(largest * cage_ratio * 3.0, 1e-4)  # normals need a slightly bigger cage
    margin = max(texture_size // 256, 4)

    scene = bpy.context.scene
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = ext
    scene.render.bake.margin = margin
    scene.render.bake.use_clear = True

    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    try:
        bpy.ops.object.bake(type="NORMAL", use_selected_to_active=True,
                            cage_extrusion=ext, margin=margin)
        log(f"bake: normal map done (extrusion={ext:.4f})")
        return img, node
    except Exception as e:
        log(f"bake: normal map FAILED: {e}")
        nt.nodes.remove(node)
        bpy.data.images.remove(img)
        return None, None


def bake_roughness_map(src, tgt, texture_size):
    """
    Bake roughness. TRELLIS output has no real material data, so this mostly
    captures the source's uniform roughness — but having the channel present,
    and letting us push metallic paint vs matte rubber later, matters more than
    the values themselves. A flat mid-roughness already beats none (no channel
    = fully rough = zero specular highlight = the "dead plastic" look).
    """
    img = bpy.data.images.new("BakedRoughness", texture_size, texture_size, alpha=False)
    img.colorspace_settings.name = "Non-Color"

    mat = tgt.data.materials[0]
    nt = mat.node_tree
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    node.name = "ROUGH_TARGET"
    nt.nodes.active = node

    margin = max(texture_size // 256, 4)
    try:
        bpy.ops.object.bake(type="ROUGHNESS", use_selected_to_active=True, margin=margin)
        log("bake: roughness map done")
        return img, node
    except Exception as e:
        log(f"bake: roughness FAILED: {e}")
        nt.nodes.remove(node)
        bpy.data.images.remove(img)
        return None, None


def wire_pbr_material(tgt, basecolor_tex, normal_img, rough_img):
    """
    Rebuild tgt's material as a proper PBR graph so the baked maps actually
    export in the GLB. Without this the normal/roughness images exist but are
    never referenced, and glTF drops them.
    """
    mat = tgt.data.materials[0]
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    base = nt.nodes.new("ShaderNodeTexImage")
    base.image = basecolor_tex.image
    nt.links.new(base.outputs["Color"], bsdf.inputs["Base Color"])

    if rough_img is not None:
        r = nt.nodes.new("ShaderNodeTexImage")
        r.image = rough_img
        r.image.colorspace_settings.name = "Non-Color"
        nt.links.new(r.outputs["Color"], bsdf.inputs["Roughness"])

    if normal_img is not None:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = normal_img
        n.image.colorspace_settings.name = "Non-Color"
        nmap = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(n.outputs["Color"], nmap.inputs["Color"])
        nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    log(f"pbr material wired: basecolor + "
        f"{'normal ' if normal_img else ''}{'roughness' if rough_img else ''}")


def export_glb(obj, path, texture_size):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Pack every baked image so they travel inside the .glb.
    for img in bpy.data.images:
        if img.name in ("BakedBaseColor", "BakedNormal", "BakedRoughness"):
            try:
                img.pack()
            except Exception:
                pass

    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_materials="EXPORT",
        export_image_format="WEBP",
        export_texture_dir="",
        export_yup=True,
        export_apply=True,
    )
    log(f"exported: {path} ({os.path.getsize(path)} bytes)")


def main():
    args = parse_args()
    log(f"input={args.input} target_faces={args.target_faces} tex={args.texture_size}")

    reset_scene()
    enable_gpu_cycles()

    src = import_glb(args.input)
    src_verts, src_faces = mesh_stats(src)

    # Keep an untouched copy as the bake source before we destroy topology.
    tgt = src.copy()
    tgt.data = src.data.copy()
    bpy.context.collection.objects.link(tgt)
    tgt.name = "RETOPO"

    clean_mesh(tgt)

    repaired, repair_method = (False, "skipped")
    if not args.no_repair:
        repaired, repair_method = make_manifold(tgt, args.repair_detail)
        log(f"repair: {'OK' if repaired else 'INCOMPLETE'} via {repair_method}")

    method = retopologize(
        tgt, args.target_faces, allow_decimate=not args.no_decimate_fallback
    )
    unwrap(tgt)

    baked = False
    bake_quality = "n/a"
    pbr_maps = "none"
    if not args.no_bake:
        try:
            bake_tex_node = make_bake_target(tgt, args.texture_size)
            bake_img_tuple = bake_tex_node
            bake_img, basecolor_node = bake_img_tuple
            lum, dark = bake_color(
                src, tgt, args.texture_size, args.bake_samples,
                bake_img, cage_ratio=args.cage_ratio,
            )
            baked = True
            bake_quality = f"luminance={lum:.1f} dark_frac={dark:.2f}"

            # PBR maps — the audit's #1 finding. Bake normal + roughness so the
            # low-poly result carries surface relief and specular response
            # instead of reading as a dead matte lump. Each is best-effort:
            # a failed map must not lose us the working baseColor.
            normal_img = rough_img = None
            if not args.no_pbr:
                normal_img, _ = bake_normal_map(
                    src, tgt, args.texture_size, cage_ratio=args.cage_ratio
                )
                rough_img, _ = bake_roughness_map(src, tgt, args.texture_size)
                wire_pbr_material(tgt, basecolor_node, normal_img, rough_img)
                pbr_maps = (("normal " if normal_img else "")
                            + ("roughness" if rough_img else "")) or "none"
        except Exception as e:
            log(f"BAKE FAILED: {e}")
            log(traceback.format_exc()[-1500:])
            log("continuing with untextured clean mesh")

    if args.smooth_shading:
        bpy.ops.object.select_all(action="DESELECT")
        tgt.select_set(True)
        bpy.context.view_layer.objects.active = tgt
        bpy.ops.object.shade_smooth()

    # Drop the dense source so only clean topology exports.
    bpy.data.objects.remove(src, do_unlink=True)

    export_glb(tgt, args.output, args.texture_size)

    out_verts, out_faces = mesh_stats(tgt)

    # Report the two properties that were previously wrong, so regressions show
    # up in the job log instead of needing an offline mesh inspection.
    boundary = count_holes(tgt)
    manifold_ok = is_manifold(tgt)[0]
    quads = tris = ngons = 0
    for poly in tgt.data.polygons:
        n = len(poly.vertices)
        if n == 4:
            quads += 1
        elif n == 3:
            tris += 1
        else:
            ngons += 1
    quad_pct = round(100.0 * quads / max(out_faces, 1), 1)

    log(
        f"DONE repair={repair_method} retopo={method} baked={baked} "
        f"{src_verts}v/{src_faces}f -> {out_verts}v/{out_faces}f | "
        f"watertight={boundary == 0} manifold={manifold_ok} "
        f"quads={quad_pct}% (q{quads}/t{tris}/n{ngons}) bake[{bake_quality}] pbr[{pbr_maps}]"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
