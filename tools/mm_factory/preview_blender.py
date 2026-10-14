"""Render a filmstrip PNG per GLB with headless Blender (Workbench), for quick visual review.

usage: blender -b --factory-startup --python tools/mm_factory/preview_blender.py -- OUT_DIR FRAMES VIEW CLIP.glb [CLIP.glb ...]
  FRAMES: number of evenly spaced frames per strip (e.g. 6)
  VIEW: side (camera on the character's left, +X) or front (camera ahead of the character); orthographic, following the Hips.
"""
import math
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out_dir, nframes, view, files = os.path.abspath(argv[0]), int(argv[1]), argv[2], [os.path.abspath(f) for f in argv[3:]]
os.makedirs(out_dir, exist_ok=True)

for glb in files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=glb)
    arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
    for o in bpy.data.objects:
        if o.type == "MESH" and len(o.data.vertices) < 100:
            o.hide_render = True  # importer's bone-shape icosphere
    act = arm.animation_data.action
    f0, f1 = (int(round(x)) for x in act.frame_range)
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "MATERIAL"
    sc.render.resolution_x, sc.render.resolution_y = 300, 420
    sc.render.film_transparent = False
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.3
    hips = arm.pose.bones.get("Hips")
    stills = []
    for i in range(nframes):
        f = f0 + round(i * (f1 - f0) / max(1, nframes - 1))
        sc.frame_set(f)
        p = arm.matrix_world @ hips.head  # follow the root (Blender Z-up)
        if view == "front":  # character faces glTF +Z = Blender -Y; camera on -Y looking toward +Y
            cam.location = (p.x, p.y - 4.0, 0.95)
            cam.rotation_euler = (math.radians(90), 0, 0)
        else:  # camera on +X (character's left side), looking toward -X
            cam.location = (p.x + 4.0, p.y, 0.95)
            cam.rotation_euler = (math.radians(90), 0, math.radians(90))
        path = os.path.join(out_dir, f"_f{i}.png")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        stills.append(path)
    # stitch with Blender's image API
    imgs = [bpy.data.images.load(s) for s in stills]
    w, h = imgs[0].size
    strip = bpy.data.images.new("strip", width=w * len(imgs), height=h)
    px = [0.0] * (w * len(imgs) * h * 4)
    for k, im in enumerate(imgs):
        src = list(im.pixels)
        for y in range(h):
            row = src[y * w * 4:(y + 1) * w * 4]
            o = (y * w * len(imgs) + k * w) * 4
            px[o:o + w * 4] = row
    strip.pixels = px
    name = os.path.splitext(os.path.basename(glb))[0]
    strip.filepath_raw = os.path.join(out_dir, f"{name}_{view}.png")
    strip.file_format = "PNG"
    strip.save()
    for s in stills:
        os.remove(s)
    print("strip", name, f"frames {f0}-{f1}")
