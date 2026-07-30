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
    p.add_argument("--no-bake", action="store_true")
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


def retopologize(obj, target_faces):
    """
    Fallback chain, best topology first:
      1. QuadriFlow  -> true quad field, needs manifold input
      2. Voxel remesh (quads) + decimate to budget
      3. Plain collapse decimate -> triangles, always works
    """
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    manifold, bad_edges = is_manifold(obj)
    log(f"manifold={manifold} (non-manifold edges: {bad_edges})")

    if manifold:
        try:
            bpy.ops.object.quadriflow_remesh(
                target_faces=target_faces,
                use_preserve_sharp=False,
                use_preserve_boundary=False,
                use_mesh_symmetry=False,
                smooth_normals=True,
            )
            log(f"retopo: quadriflow -> {mesh_stats(obj)[1]} faces")
            return "quadriflow"
        except Exception as e:
            log(f"quadriflow failed: {e}")

    # Voxel remesh produces a clean manifold quad-ish surface from messy input.
    try:
        dims = obj.dimensions
        largest = max(dims.x, dims.y, dims.z) or 1.0
        # Aim for roughly sqrt(target_faces) quads across the widest axis.
        voxel = largest / max(int((target_faces ** 0.5) * 1.5), 32)
        obj.data.remesh_voxel_size = max(voxel, 1e-4)
        obj.data.remesh_voxel_adaptivity = 0.0
        bpy.ops.object.voxel_remesh()
        log(f"retopo: voxel_remesh @ {obj.data.remesh_voxel_size:.5f} -> {mesh_stats(obj)[1]} faces")

        cur = mesh_stats(obj)[1]
        if cur > target_faces * 1.1:
            m = obj.modifiers.new("dec", "DECIMATE")
            m.decimate_type = "COLLAPSE"
            m.ratio = target_faces / cur
            bpy.ops.object.modifier_apply(modifier=m.name)
            log(f"retopo: + decimate -> {mesh_stats(obj)[1]} faces")
        return "voxel_remesh"
    except Exception as e:
        log(f"voxel remesh failed: {e}")

    cur = mesh_stats(obj)[1]
    if cur > target_faces:
        m = obj.modifiers.new("dec", "DECIMATE")
        m.decimate_type = "COLLAPSE"
        m.ratio = target_faces / cur
        bpy.ops.object.modifier_apply(modifier=m.name)
    log(f"retopo: decimate only -> {mesh_stats(obj)[1]} faces")
    return "decimate"


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


def bake_color(src, tgt, texture_size, samples):
    """Bake SOURCE's base color onto TGT's new UVs via selected_to_active."""
    scene = bpy.context.scene
    scene.cycles.samples = max(samples, 1)
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.use_cage = False
    scene.render.bake.cage_extrusion = 0.02
    scene.render.bake.margin = max(texture_size // 256, 4)
    scene.render.bake.use_clear = True

    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt  # active = bake destination

    bpy.ops.object.bake(
        type="DIFFUSE",
        pass_filter={"COLOR"},
        use_selected_to_active=True,
        cage_extrusion=0.02,
        margin=scene.render.bake.margin,
    )
    log("bake: diffuse/color complete")


def export_glb(obj, path, texture_size):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Pack the baked image so it travels inside the .glb.
    for img in bpy.data.images:
        if img.name == "BakedBaseColor":
            img.pack()

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
    method = retopologize(tgt, args.target_faces)
    unwrap(tgt)

    baked = False
    if not args.no_bake:
        try:
            make_bake_target(tgt, args.texture_size)
            bake_color(src, tgt, args.texture_size, args.bake_samples)
            baked = True
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
    log(
        f"DONE retopo={method} baked={baked} "
        f"{src_verts}v/{src_faces}f -> {out_verts}v/{out_faces}f"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        sys.exit(1)
