"""
Silicon Neuron — Pepper's-ghost demo enclosure, built entirely from code.

Open in Blender (4.2+ / 5.x):  Scripting tab -> Open -> build_demo.py -> Run Script
or from a terminal:            blender -b -P build_demo.py -- --render
or with the pip 'bpy' module:  python build_demo.py --render

Everything is parametric: change a dimension in PARAMETERS and re-run. The
script also writes cut_sheet.svg (1:1 panel templates for 5 mm foam board or
3 mm MDF) from the same numbers, so the model and the cut parts never disagree.

Coordinate system (millimetres): x to the viewer's right, y AWAY from the
viewer (front opening at y = 0), z up from the table.

Optics (why the parts are where they are)
-----------------------------------------
The 7" screen lies FACE-DOWN in the roof. Under it a clear sheet sits at 45
degrees, rising from the front-bottom (y = PANE_Y0, z = 0) to the back of the
roof (y = PANE_Y0 + IN_H, z = IN_H). Reflection in the plane y - z = PANE_Y0
maps a screen point (y0, z = IN_H) to (y = IN_H + PANE_Y0, z = y0 - PANE_Y0):
every pixel's reflection lands on ONE vertical plane at y = IN_H + PANE_Y0, the
"ghost plane". Screen points further back appear higher, so the display's top
edge must face the back of the box, and the image is mirrored left-right in
software (hologram.py does both).
"""
import math
import os
import sys

# ============================================================== PARAMETERS
T_WALL = 5.0          # foam board / MDF thickness
IN_W = 196.0          # interior width
IN_D = 220.0          # interior depth (front opening -> back wall)
IN_H = 120.0          # viewing chamber height (floor -> underside of roof)
CAP_H = 45.0          # electronics compartment above the roof
PANE_Y0 = 20.0        # where the 45-degree pane meets the floor
PANE_W = 190.0        # acrylic sheet width
PANE_T = 2.0          # acrylic thickness (2 mm gives less double-image than 3 mm)
VALANCE_H = 8.0       # lip hanging from the roof front: hides the screen from low viewers

# 7" 1024x600 HDMI IPS (e.g. Waveshare 7inch HDMI LCD (C)) — check your datasheet
SCR_W, SCR_D, SCR_T = 164.9, 100.0, 5.0
ACT_W, ACT_D = 154.2, 85.9          # active (lit) area
DRV_W, DRV_D, DRV_T = 110.0, 60.0, 11.0

# derived
GHOST_Y = IN_H + PANE_Y0            # the reflection stands on this plane
PANE_L = IN_H * math.sqrt(2)        # sheet length along the slope
ACT_Y1 = GHOST_Y                    # back edge of the lit area sits right above the ghost plane
ACT_Y0 = ACT_Y1 - ACT_D
EXT_W = IN_W + 2 * T_WALL
EXT_D = IN_D + T_WALL
EXT_H = T_WALL + IN_H + T_WALL + CAP_H + T_WALL
SLOT_X, SLOT_TOP, SLOT_W, SLOT_H = 20.0, T_WALL + 10, 24.0, 12.0   # back-wall cable slot (from top-left, seen from the front)
VENT_X0, VENT_PITCH, VENT_W, VENT_Y0, VENT_D = 30.0, 22.0, 10.0, IN_D - 40, 30.0   # 6 lid vents

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


# ============================================================== cut sheet (no bpy needed)
def write_cut_sheet(path):
    """1:1 SVG templates. Print at 100% (or send to a laser cutter)."""
    panels = [
        # name, width, height, cut-outs [(kind, x, y, w, h | r)]  (x, y from panel's top-left)
        ("FLOOR (black)", IN_W, IN_D, []),
        ("ROOF (screen window; top edge of drawing = front)", IN_W, IN_D,
         [("rect", (IN_W - ACT_W) / 2, ACT_Y0, ACT_W, ACT_D), ("circle", IN_W - 26, IN_D - 26, 7)]),
        ("SIDE L", EXT_D, EXT_H, []),
        ("SIDE R", EXT_D, EXT_H, []),
        ("BACK, seen from the front (slot = USB power in, Pi side)", IN_W, EXT_H,
         [("rect", SLOT_X, SLOT_TOP, SLOT_W, SLOT_H)]),
        ("CAP FRONT (24 mm button hole)", IN_W, CAP_H, [("circle", IN_W - 30, CAP_H / 2, 12)]),
        ("CAP LID (top edge of drawing = front)", IN_W, IN_D,
         [("rect", VENT_X0 + i * VENT_PITCH, VENT_Y0, VENT_W, VENT_D) for i in range(6)]),
        ("VALANCE", IN_W, VALANCE_H, []),
        ("PANE RAIL x4 (glue at 45 deg)", PANE_L, 8, []),
        ("ACRYLIC PANE 2 mm (order cut)", PANE_W, PANE_L, []),
    ]
    x, y, row_h, sheet_w = 10, 10, 0, 620
    items = []
    for name, w, h, cuts in panels:
        if x + w > sheet_w:
            x, y, row_h = 10, y + row_h + 25, 0
        items.append(f'<rect x="{x}" y="{y}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="red" stroke-width="0.3"/>')
        items.append(f'<text x="{x + 3}" y="{y + 9}" font-size="6" font-family="sans-serif">{name}  {w:.0f} x {h:.0f} mm</text>')
        for c in cuts:
            if c[0] == "rect" and c[3] > 0.01:
                items.append(f'<rect x="{x + c[1]:.1f}" y="{y + c[2]:.1f}" width="{c[3]:.1f}" height="{c[4]:.1f}" '
                             f'fill="none" stroke="blue" stroke-width="0.3"/>')
            elif c[0] == "circle":
                items.append(f'<circle cx="{x + c[1]:.1f}" cy="{y + c[2]:.1f}" r="{c[3]:.1f}" fill="none" '
                             f'stroke="blue" stroke-width="0.3"/>')
        x += w + 10
        row_h = max(row_h, h)
    H = y + row_h + 20
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{sheet_w}mm" height="{H:.0f}mm" '
           f'viewBox="0 0 {sheet_w} {H:.0f}">\n<rect width="100%" height="100%" fill="white"/>\n'
           + "\n".join(items) +
           f'\n<text x="10" y="{H - 6:.0f}" font-size="5" font-family="sans-serif">red = outline, blue = cut-out. '
           f'Check the 100 mm scale bar before cutting.</text>\n'
           f'<line x1="400" y1="{H - 8:.0f}" x2="500" y2="{H - 8:.0f}" stroke="black" stroke-width="0.8"/>'
           f'<text x="440" y="{H - 11:.0f}" font-size="5">100 mm</text>\n</svg>\n')
    with open(path, "w") as f:
        f.write(svg)
    print("wrote", path)


# ============================================================== Blender part
def build():
    import bpy
    import bmesh
    from mathutils import Vector, Euler, Matrix

    MM = 0.001                                 # model in metres, dimensions in mm

    # ---------------------------------------------------------------- reset
    # wipe everything this script (or the default scene) made, without reloading
    # the file, so it is safe to run from Blender's own Text Editor
    for data in (bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                 bpy.data.lights, bpy.data.cameras, bpy.data.images, bpy.data.worlds):
        for block in list(data):
            data.remove(block)
    scene = bpy.context.scene
    for c in list(scene.collection.children):
        bpy.data.collections.remove(c)
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"

    cols = {}

    def col(name):
        if name not in cols:
            c = bpy.data.collections.new(name)
            scene.collection.children.link(c)
            cols[name] = c
        return cols[name]

    # ---------------------------------------------------------------- materials
    def mat(name, color, metallic=0.0, rough=0.5, emission=None, strength=0.0,
            transmission=0.0, ior=1.45, alpha=1.0, image=None):
        m = bpy.data.materials.new(name)
        m.diffuse_color = (*(emission if (emission and strength > 1) else color), alpha)  # Workbench / viewport colour
        m.metallic, m.roughness = metallic, rough
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes.get("Principled BSDF")

        def setin(key, val):
            for k in ([key] if isinstance(key, str) else key):
                if k in b.inputs:
                    b.inputs[k].default_value = val
                    return
        setin("Base Color", (*color, 1.0))
        setin("Metallic", metallic)
        setin("Roughness", rough)
        setin(["Transmission Weight", "Transmission"], transmission)
        setin("IOR", ior)
        setin("Alpha", alpha)
        if emission is not None:
            setin(["Emission Color", "Emission"], (*emission, 1.0))
            setin("Emission Strength", strength)
        if image is not None:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(image)
            nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
            ek = "Emission Color" if "Emission Color" in b.inputs else "Emission"
            nt.links.new(tex.outputs["Color"], b.inputs[ek])
            setin("Emission Strength", strength)
        if alpha < 1.0:
            try:
                m.surface_render_method = "BLENDED"
            except AttributeError:
                m.blend_method = "BLEND"
        return m

    M = {
        "board": mat("foam board (matt black)", (0.012, 0.012, 0.014), rough=0.9),
        "cap": mat("smoked acrylic cap", (0.02, 0.02, 0.025), rough=0.05, transmission=0.0, alpha=0.35),
        "acrylic": mat("clear acrylic pane", (1, 1, 1), rough=0.0, transmission=1.0, ior=1.49),
        "bezel": mat("screen bezel", (0.03, 0.03, 0.03), rough=0.4),
        "pcb_g": mat("PCB green", (0.02, 0.25, 0.08), rough=0.45),
        "pcb_p": mat("PCB purple", (0.18, 0.05, 0.3), rough=0.45),
        "pcb_b": mat("PCB blue", (0.03, 0.1, 0.35), rough=0.45),
        "chip": mat("IC black", (0.01, 0.01, 0.01), rough=0.3),
        "metal": mat("tin plate", (0.8, 0.8, 0.82), metallic=1.0, rough=0.25),
        "gold": mat("pins", (0.9, 0.7, 0.3), metallic=1.0, rough=0.3),
        "red": mat("wire red", (0.7, 0.03, 0.03), rough=0.5),
        "blk": mat("wire black", (0.02, 0.02, 0.02), rough=0.5),
        "yel": mat("wire yellow", (0.8, 0.6, 0.02), rough=0.5),
        "grn": mat("wire green", (0.05, 0.5, 0.1), rough=0.5),
        "blu": mat("wire blue", (0.05, 0.2, 0.8), rough=0.5),
        "wht": mat("wire white", (0.8, 0.8, 0.8), rough=0.5),
        "org": mat("wire orange", (0.95, 0.35, 0.02), rough=0.5),
        "gry": mat("wire grey", (0.4, 0.4, 0.42), rough=0.5),
        "pur": mat("wire purple", (0.4, 0.05, 0.6), rough=0.5),
        "term": mat("terminal block", (0.05, 0.35, 0.6), rough=0.5),
        "cable": mat("cable grey", (0.08, 0.08, 0.09), rough=0.6),
        "button": mat("arcade button", (0.9, 0.1, 0.5), rough=0.3, emission=(1, 0.2, 0.6), strength=2.0),
        "cyan": mat("cyan glow", (0, 0.9, 1), emission=(0, 0.9, 1), strength=6.0),
        "led": mat("led", (0.1, 1, 0.3), emission=(0.1, 1, 0.3), strength=30.0),
        "table": mat("table", (0.25, 0.2, 0.16), rough=0.6),
        "ray_in": mat("ray from screen", (1, 0.25, 0.8), emission=(1, 0.25, 0.8), strength=8.0),
        "ray_out": mat("ray to eye", (0, 0.9, 1), emission=(0, 0.9, 1), strength=8.0),
        "ray_virt": mat("virtual ray", (1, 1, 1), emission=(1, 1, 1), strength=3.0),
        "pane_x": mat("pane (exploded view tint)", (0.5, 0.85, 1), rough=0.1, emission=(0.3, 0.7, 1),
                      strength=0.4, alpha=0.35),
        "label": mat("label", (1, 1, 1), emission=(1, 1, 1), strength=4.0),
        "ghost": mat("ghost plane (diagram only)", (0.2, 0.9, 1), emission=(0.2, 0.9, 1), strength=0.6, alpha=0.18),
    }
    frame = os.path.join(HERE, "screen_frame.png")
    M["screen"] = mat("screen image", (0, 0, 0), emission=(1, 1, 1), strength=30.0,
                      image=frame if os.path.exists(frame) else None)

    # ---------------------------------------------------------------- mesh helpers
    def box(name, size, loc, material, collection="Enclosure", bevel=0.0, rot=(0, 0, 0)):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co.x *= size[0] * MM
            v.co.y *= size[1] * MM
            v.co.z *= size[2] * MM
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(name, me)
        ob.location = Vector(loc) * MM
        ob.rotation_euler = Euler(rot)
        ob.data.materials.append(material)
        col(collection).objects.link(ob)
        if bevel:
            md = ob.modifiers.new("bevel", "BEVEL")
            md.width = bevel * MM
            md.segments = 3
        return ob

    def box_min(name, x0, y0, z0, x1, y1, z1, material, collection="Enclosure", **kw):
        """Box from min/max corners (mm)."""
        return box(name, (x1 - x0, y1 - y0, z1 - z0),
                   ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), material, collection, **kw)

    def cyl(name, r, h, loc, material, collection="Electronics", rot=(0, 0, 0), seg=32):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r * MM, radius2=r * MM, depth=h * MM)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(name, me)
        ob.location = Vector(loc) * MM
        ob.rotation_euler = Euler(rot)
        ob.data.materials.append(material)
        col(collection).objects.link(ob)
        return ob

    def plane_uv(name, w, d, loc, material, collection, rot=(0, 0, 0)):
        me = bpy.data.meshes.new(name)
        hw, hd = w * MM / 2, d * MM / 2
        me.from_pydata([(-hw, -hd, 0), (hw, -hd, 0), (hw, hd, 0), (-hw, hd, 0)], [], [(0, 1, 2, 3)])
        uv = me.uv_layers.new(name="UVMap")
        for i, co in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
            uv.data[i].uv = co
        ob = bpy.data.objects.new(name, me)
        ob.location = Vector(loc) * MM
        ob.rotation_euler = Euler(rot)
        ob.data.materials.append(material)
        col(collection).objects.link(ob)
        return ob

    def wire(name, pts, material, radius=0.9, collection="Wiring"):
        cu = bpy.data.curves.new(name, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = radius * MM
        cu.bevel_resolution = 3
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(pts) - 1)
        for bp, p in zip(sp.bezier_points, pts):
            bp.co = Vector(p) * MM
            bp.handle_left_type = bp.handle_right_type = "AUTO"
        ob = bpy.data.objects.new(name, cu)
        ob.data.materials.append(material)
        col(collection).objects.link(ob)
        return ob

    def rod(name, p0, p1, material, r=0.8, collection="Diagram"):
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        ob = cyl(name, r, d.length, (p0 + p1) / 2, material, collection)
        ob.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        return ob

    def text(name, body, loc, size, material, collection="Labels", rot=(math.pi / 2, 0, 0), extrude=0.0,
             align="CENTER"):
        cu = bpy.data.curves.new(name, "FONT")
        cu.body = body
        cu.size = size * MM
        cu.align_x = align
        cu.align_y = "CENTER"
        cu.extrude = extrude * MM
        ob = bpy.data.objects.new(name, cu)
        ob.location = Vector(loc) * MM
        ob.rotation_euler = Euler(rot)
        ob.data.materials.append(material)
        col(collection).objects.link(ob)
        return ob

    parts = {}          # name -> (object list, exploded offset in mm)

    def part(key, objs, offset):
        parts.setdefault(key, ([], offset))[0].extend(objs if isinstance(objs, list) else [objs])

    X0, X1 = -IN_W / 2, IN_W / 2
    Z_ROOF = T_WALL + IN_H                  # underside of roof
    Z_CAP = Z_ROOF + T_WALL                 # top of roof / floor of cap

    # ---------------------------------------------------------------- enclosure
    part("floor", box_min("Floor", X0, 0, 0, X1, IN_D, T_WALL, M["board"]), (0, 0, -50))
    part("side L", box_min("Side L", X0 - T_WALL, 0, 0, X0, EXT_D, EXT_H, M["board"]), (-150, 0, 0))
    part("side R", box_min("Side R", X1, 0, 0, X1 + T_WALL, EXT_D, EXT_H, M["board"]), (150, 0, 0))
    # back wall with the 24 x 12 cable slot at the top, on the Pi's side (viewer's left)
    sx0, sx1 = X0 + SLOT_X, X0 + SLOT_X + SLOT_W
    sz1 = EXT_H - SLOT_TOP
    sz0 = sz1 - SLOT_H
    part("back", [box_min("Back", X0, IN_D, 0, X1, EXT_D, sz0, M["board"]),
                  box_min("Back top", X0, IN_D, sz1, X1, EXT_D, EXT_H, M["board"]),
                  box_min("Back slot left", X0, IN_D, sz0, sx0, EXT_D, sz1, M["board"]),
                  box_min("Back slot right", sx1, IN_D, sz0, X1, EXT_D, sz1, M["board"])], (0, 150, 0))
    # roof = four strips around the lit-area window (no booleans needed)
    wx0, wx1 = -ACT_W / 2, ACT_W / 2
    roof = [box_min("Roof front", X0, 0, Z_ROOF, X1, ACT_Y0, Z_CAP, M["board"]),
            box_min("Roof back", X0, ACT_Y1, Z_ROOF, X1, IN_D, Z_CAP, M["board"]),
            box_min("Roof left", X0, ACT_Y0, Z_ROOF, wx0, ACT_Y1, Z_CAP, M["board"]),
            box_min("Roof right", wx1, ACT_Y0, Z_ROOF, X1, ACT_Y1, Z_CAP, M["board"])]
    part("roof", roof, (0, 0, 75))
    part("valance", box_min("Valance", X0, 0, Z_ROOF - VALANCE_H, X1, T_WALL, Z_ROOF, M["board"]), (0, -120, 75))
    # lid: strips around the 6 vent slots over the Pi and meter (names all start "Cap lid")
    lz0, lz1 = Z_CAP + CAP_H, Z_CAP + CAP_H + T_WALL
    vy0, vy1 = VENT_Y0, VENT_Y0 + VENT_D
    lid = [box_min("Cap lid", X0, 0, lz0, X1, vy0, lz1, M["cap"], "Cap"),
           box_min("Cap lid back", X0, vy1, lz0, X1, IN_D, lz1, M["cap"], "Cap")]
    xs = [X0] + [v for i in range(6) for v in (X0 + VENT_X0 + i * VENT_PITCH,
                                               X0 + VENT_X0 + i * VENT_PITCH + VENT_W)] + [X1]
    for i in range(0, len(xs), 2):
        lid.append(box_min(f"Cap lid bar {i // 2}", xs[i], vy0, lz0, xs[i + 1], vy1, lz1, M["cap"], "Cap"))
    cap = [box_min("Cap front", X0, 0, Z_CAP, X1, T_WALL, Z_CAP + CAP_H, M["cap"], "Cap")] + lid
    part("cap", cap, (0, 0, 250))

    # pane rails: small strips on the side walls holding the sheet at 45 degrees
    rails = []
    for sx in (X0 + 2.5, X1 - 2.5):
        for off in (-PANE_T / 2 - 2.5, PANE_T / 2 + 2.5):
            yc = PANE_Y0 + IN_H / 2 - off / math.sqrt(2)
            zc = T_WALL + IN_H / 2 + off / math.sqrt(2)
            rails.append(box(f"Rail {sx:.0f} {off:.0f}", (5, PANE_L, 3), (sx, yc, zc), M["board"],
                             rot=(math.pi / 4, 0, 0)))
    part("rails", [r for r in rails if r.location.x < 0], (-150, 0, 0))
    part("rails R", [r for r in rails if r.location.x > 0], (150, 0, 0))

    # the 45-degree pane
    pane = box("Acrylic pane 45deg", (PANE_W, PANE_L, PANE_T),
               (0, PANE_Y0 + IN_H / 2, T_WALL + IN_H / 2), M["acrylic"], "Optics", rot=(math.pi / 4, 0, 0))
    part("pane", pane, (0, -190, 20))

    # ---------------------------------------------------------------- display (face down in the roof)
    scr_cy = (ACT_Y0 + ACT_Y1) / 2
    bezel = box("7in display", (SCR_W, SCR_D, SCR_T), (0, scr_cy, Z_CAP + SCR_T / 2), M["bezel"], "Display")
    # lit face: plane facing DOWN (rotated 180 deg about y keeps left/right = mirrored-image convention)
    lit = plane_uv("Display lit area", ACT_W, ACT_D, (0, scr_cy, Z_CAP - 0.2), M["screen"], "Display",
                   rot=(0, math.pi, 0))
    drv = box("HDMI driver board", (DRV_W, DRV_D, DRV_T - 2), (0, scr_cy, Z_CAP + SCR_T + DRV_T / 2), M["pcb_b"],
              "Display")
    part("display", [bezel, lit, drv], (0, 0, 150))

    # ---------------------------------------------------------------- Raspberry Pi Zero 2 W in the cap
    # 2 x 20 header along the back edge, ports facing the front. Pin 1 is at the left
    # end; odd pins in the inner row, even pins (5 V, GND, TXD ...) on the board edge.
    pz = []
    pzx, pzy, pzz = -45.0, 190.0, Z_CAP + 3              # on ~2 mm foam tape
    hdr_y = pzy + 11.5

    def pi_pin(n):
        """Top of Raspberry Pi header pin n (1..40), mm."""
        return ((pzx - 24.13 + ((n - 1) // 2) * 2.54), hdr_y + (1.27 if n % 2 == 0 else -1.27), pzz + 15)
    pz.append(box("Pi Zero 2 W", (65, 30, 1.4), (pzx, pzy, pzz), M["pcb_g"], "Electronics", bevel=1.0))
    pz.append(box("Pi Zero SoC", (12, 12, 1.2), (pzx - 5, pzy, pzz + 1.3), M["chip"], "Electronics"))
    pz.append(box("Pi Zero header", (51, 5, 8.5), (pzx, hdr_y, pzz + 5), M["chip"], "Electronics"))
    pz.append(box("Pi pin 1 marker", (2.2, 0.6, 0.2), (pi_pin(1)[0], hdr_y - 3.4, pzz + 0.8), M["wht"],
                  "Electronics"))
    for n in range(1, 41):
        x, y, _ = pi_pin(n)
        pz.append(box(f"pz pin {n}", (0.64, 0.64, 6), (x, y, pzz + 12), M["gold"], "Electronics"))
    pz.append(box("mini HDMI", (11, 7.5, 3.3), (pzx - 20, pzy - 14, pzz + 2.4), M["metal"], "Electronics"))
    pz.append(box("micro USB OTG", (8, 5.6, 2.6), (pzx + 9, pzy - 14.5, pzz + 2.1), M["metal"], "Electronics"))
    pz.append(box("micro USB pwr", (8, 5.6, 2.6), (pzx + 22, pzy - 14.5, pzz + 2.1), M["metal"], "Electronics"))
    part("pi", pz, (0, -260, 200))

    # INA226 module (in the cap next to the Pi: short 5 V path). Logic header on the
    # left edge (VCC, GND, SCL, SDA from front to back), VIN+/VIN- screw terminal on the right.
    inz = Z_CAP + 1.8
    INA = {"VCC": (24.5, 186.19), "GND": (24.5, 188.73), "SCL": (24.5, 191.27), "SDA": (24.5, 193.81),
           "VIN+": (53.5, 186.5), "VIN-": (53.5, 193.5)}
    ina = [box("INA226 module", (36, 21, 1.6), (40, 190, inz), M["pcb_p"], "Electronics", bevel=0.8),
           box("INA226 IC", (3, 3, 1), (38, 190, inz + 1.3), M["chip"], "Electronics"),
           box("shunt R100", (6.3, 3.2, 0.8), (44, 184, inz + 1.2), M["chip"], "Electronics"),
           box("INA226 terminal", (7, 13, 7), (53.5, 190, inz + 4.3), M["term"], "Electronics")]
    for k in ("VCC", "GND", "SCL", "SDA"):
        ina.append(box(f"INA226 pin {k}", (0.64, 0.64, 7), (*INA[k], inz + 3.5), M["gold"], "Electronics"))
    ina_top = {k: (x, y, inz + 7 if k.startswith("VIN") else inz + 6.5) for k, (x, y) in INA.items()}
    part("ina226", ina, (20, -260, 200))

    # Arcade button on the cap front; its switch body and two legs stick out behind the panel
    bx_, bz_ = IN_W / 2 - 30, Z_CAP + CAP_H / 2
    btn = [cyl("Button bezel", 12, 4, (bx_, -2, bz_), M["chip"], rot=(math.pi / 2, 0, 0)),
           cyl("Button cap", 8, 5, (bx_, -5, bz_), M["button"], rot=(math.pi / 2, 0, 0)),
           cyl("Button body", 11, 18, (bx_, T_WALL + 9, bz_), M["chip"], rot=(math.pi / 2, 0, 0)),
           box("Button switch", (12, 8, 10), (bx_, T_WALL + 22, bz_), M["chip"])]
    legs = [(bx_ - 3.5, T_WALL + 29, bz_), (bx_ + 3.5, T_WALL + 29, bz_)]
    for k, lp in enumerate(legs):
        btn.append(box(f"Button leg {k}", (2.5, 6, 0.6), lp, M["metal"]))
    part("button", btn, (0, -60, 250))

    # ---------------------------------------------------------------- the mint tin + Pico behind the ghost
    TIN_W, TIN_D, TIN_H = 95.0, 61.0, 21.0
    ty0 = GHOST_Y + 10
    tcx = 0.0
    tw = 0.8                                   # tin wall thickness: an open tray, so its contents show
    tin = [box("Mint tin floor", (TIN_W, TIN_D, tw), (tcx, ty0 + TIN_D / 2, T_WALL + tw / 2), M["metal"], "Tin"),
           box("Mint tin wall front", (TIN_W, tw, TIN_H), (tcx, ty0 + tw / 2, T_WALL + TIN_H / 2), M["metal"], "Tin"),
           box("Mint tin wall back", (TIN_W, tw, TIN_H), (tcx, ty0 + TIN_D - tw / 2, T_WALL + TIN_H / 2), M["metal"], "Tin"),
           box("Mint tin wall left", (tw, TIN_D, TIN_H), (tcx - TIN_W / 2 + tw / 2, ty0 + TIN_D / 2, T_WALL + TIN_H / 2),
               M["metal"], "Tin"),
           box("Mint tin wall right", (tw, TIN_D, TIN_H), (tcx + TIN_W / 2 - tw / 2, ty0 + TIN_D / 2, T_WALL + TIN_H / 2),
               M["metal"], "Tin")]
    lid = box("Mint tin lid", (TIN_W + 1, TIN_D + 1, 6), (0, 0, 0), M["metal"], "Tin", bevel=6.0)
    # hinge on the back edge, opened ~95 degrees
    hinge = Vector((tcx, ty0 + TIN_D, T_WALL + TIN_H)) * MM
    lid.location = hinge + Vector((0, 0, 0))
    lid.data.transform(Matrix.Translation(Vector((0, -(TIN_D + 1) / 2, 3)) * MM))
    lid.rotation_euler = Euler((-math.radians(95), 0, 0))
    tin.append(lid)
    tin.append(text("Tin label", "SILICON\nNEURON", (0, ty0 + TIN_D + 32 * math.tan(math.radians(5)) - 0.6, T_WALL + TIN_H + 32), 9, M["cyan"],
                    "Tin", rot=(math.radians(85), 0, 0), extrude=0.3))
    part("tin", tin, (340, -150, 0))

    # Half-size (400-point) breadboard filling the tin; the Pico plugs into it
    # straddling the centre channel. Rows 1-30 run along x, columns a-j across y
    # (a at the front), a pair of power rails along each long side.
    BB_W, BB_D, BB_H = 82.5, 54.5, 8.5
    bbx, bby = 0.0, ty0 + TIN_D / 2
    bb_top = T_WALL + tw + BB_H
    row_x = lambda r: bbx - 36.83 + (r - 1) * 2.54                  # r = 1..30
    COL = dict(zip("abcdefghij", (-13.97, -11.43, -8.89, -6.35, -3.81, 3.81, 6.35, 8.89, 11.43, 13.97)))
    col_y = lambda c: bby + COL[c]
    RAIL_Y = {"front -": -21.59, "front +": -19.05, "back +": 19.05, "back -": 21.59}
    rail_y = lambda r: bby + RAIL_Y[r]
    M["bb"] = mat("breadboard white", (0.9, 0.9, 0.87), rough=0.6)
    M["rail_r"] = mat("rail red", (0.8, 0.1, 0.1), rough=0.5)
    M["rail_b"] = mat("rail blue", (0.1, 0.2, 0.8), rough=0.5)
    bb = [box("Half-size breadboard (400 points)", (BB_W, BB_D, BB_H), (bbx, bby, bb_top - BB_H / 2), M["bb"], "Tin",
              bevel=0.6),
          box("Breadboard centre channel", (BB_W - 4, 2.6, 0.8), (bbx, bby, bb_top - 0.3), M["chip"], "Tin")]
    me = bpy.data.meshes.new("breadboard holes")
    bmh = bmesh.new()

    def hole_at(x, y):
        res = bmesh.ops.create_cube(bmh, size=1.0)
        for v in res["verts"]:
            v.co.x = v.co.x * 1.0 * MM + x * MM
            v.co.y = v.co.y * 1.0 * MM + y * MM
            v.co.z = v.co.z * 0.6 * MM + (bb_top - 0.2) * MM
    for r in range(1, 31):
        for c in "abcdefghij":
            hole_at(row_x(r), col_y(c))
    for rn in RAIL_Y:
        for g in range(5):                       # 5 groups of 5 holes per rail
            for h in range(5):
                hole_at(bbx - 33.0 + g * 15.24 + h * 2.54, rail_y(rn))
    bmh.to_mesh(me)
    bmh.free()
    holes = bpy.data.objects.new("Breadboard holes", me)
    holes.data.materials.append(M["chip"])
    col("Tin").objects.link(holes)
    bb.append(holes)
    for rn, y in RAIL_Y.items():
        stripe_y = bby + y + (1.6 if y > 0 else -1.6) * (1 if "-" in rn else -1)
        bb.append(box(f"rail stripe {rn}", (BB_W - 8, 0.6, 0.2), (bbx, stripe_y, bb_top + 0.05),
                      M["rail_b"] if "-" in rn else M["rail_r"], "Tin"))
    # printed row numbers (front and back edge) and column letters (left end), like the real board
    M["print"] = mat("breadboard print", (0.15, 0.15, 0.15), rough=0.6)
    flat = (0, 0, 0)
    for r in (1, 5, 10, 15, 20, 25, 30):
        for yy in (col_y("a") - 2.6, col_y("j") + 2.6):
            bb.append(text(f"bb row {r} {yy:.0f}", str(r), (row_x(r), yy, bb_top + 0.06), 1.9, M["print"], "Tin",
                           rot=flat))
    for c in "abcdefghij":
        bb.append(text(f"bb col {c}", c, (row_x(1) - 3.0, col_y(c), bb_top + 0.06), 1.9, M["print"], "Tin",
                       rot=flat))
    for rn in RAIL_Y:
        bb.append(text(f"bb rail {rn}", "+" if "+" in rn else "-", (bbx - 38.3, rail_y(rn), bb_top + 0.06), 2.4,
                       M["rail_r"] if "+" in rn else M["rail_b"], "Tin", rot=flat))
    part("breadboard", bb, (340, -150, 0))

    # Pico on rows 1-20, USB end to the left; pins in column c (front, pins 1-20)
    # and column h (back, pins 40..21). Pin 1 = GP0 at row 1, front.
    pc = []
    pcx, pcy, pcz = row_x(10.5), bby, bb_top + 3.0
    pc.append(box("Raspberry Pi Pico", (51, 21, 1.0), (pcx, pcy, pcz), M["pcb_g"], "Tin", bevel=0.8))
    pc.append(box("RP2040", (7, 7, 1), (pcx + 2, pcy, pcz + 1), M["chip"], "Tin"))
    pc.append(box("Pico USB", (8, 6, 3), (pcx - 24, pcy, pcz + 2), M["metal"], "Tin"))
    pc.append(box("Pico LED", (1.5, 1, 0.6), (pcx - 17, pcy + 6, pcz + 0.9), M["led"], "Tin"))
    for c in "ch":
        pc.append(box(f"Pico header {c}", (50.8, 2.5, 2.5), (pcx, col_y(c), bb_top + 1.25), M["chip"], "Tin"))
        for r in range(1, 21):
            pc.append(box(f"pico pin {c}{r}", (0.64, 0.64, 8), (row_x(r), col_y(c), bb_top + 1), M["gold"], "Tin"))
    part("pico", pc, (340, -150, 0))

    # 1N5819: anode in the back + rail at row 6, cathode (band) in j2 -> same strip as h2 = pin 39, VSYS
    M["diode_band"] = mat("diode band", (0.85, 0.85, 0.85), metallic=0.5, rough=0.3)
    pa = Vector((row_x(6), rail_y("back +"), bb_top))
    pk = Vector((row_x(2), col_y("j"), bb_top))
    mid = (pa + pk) / 2 + Vector((0, 0, 4))
    dvec = pk - pa
    body = cyl("1N5819 body", 1.35, 5.2, mid, M["chip"], "Tin")
    body.rotation_euler = dvec.to_track_quat("Z", "Y").to_euler()
    band = cyl("1N5819 band", 1.42, 0.9, mid + dvec.normalized() * 2.0, M["diode_band"], "Tin")
    band.rotation_euler = body.rotation_euler.copy()
    dio = [body, band,
           wire("1N5819 leads", [tuple(pa), tuple(pa + Vector((0, 0, 4))), tuple(mid), tuple(pk + Vector((0, 0, 4))),
                                 tuple(pk)], M["metal"], radius=0.3, collection="Tin")]
    part("diode", dio, (340, -150, 0))

    # breadboard holes the Pi's wires land in
    LAND = {"5V": (row_x(10), rail_y("back +")), "GND": (row_x(3), col_y("a")), "TX": (row_x(2), col_y("a")),
            "RX": (row_x(1), col_y("a")), "MARKER": (row_x(4), col_y("a"))}

    # ---------------------------------------------------------------- wiring (Pi header -> roof hole -> tin)
    hole = (IN_W / 2 - 26, IN_D - 26)
    grom = cyl("Grommet", 7, T_WALL + 1, (hole[0], hole[1], Z_ROOF + T_WALL / 2), M["chip"], "Enclosure")
    part("roof", grom, (0, 0, 75))
    # The five Pico jumpers are drawn from the tin END (index 0) up to the Pi, so the
    # animation can show them plugged in first (step 9), fed through the roof (step 12)
    # and finally connected (step 18). Control points, tin end first:
    #   0 hole, 1 just above, 2 arc over the tin, 3 LOOSE END gathered at the back right,
    #   4 under the roof hole, 5 just above the roof, 6 arc over the meter, 7 the Pi / meter pin
    colours = [("5V", "red", None), ("GND", "blk", 9), ("TX", "yel", 8), ("RX", "grn", 10), ("MARKER", "blu", 11)]
    WIRE_LOOSE, WIRE_ROOF = 3, 5               # control-point indices read by assembly_animation.py
    wires = []
    for k, (nm, cm, pin) in enumerate(colours):
        off = (k - 2) * 1.8
        # the Pico's 5 V comes out of the meter's VIN-, never straight from the Pi
        start = ina_top["VIN-"] if pin is None else pi_pin(pin)
        ex, ey = LAND[nm]
        pts = [(ex, ey, bb_top + 0.5), (ex, ey, bb_top + 7),
               (ex + 4, ey + (6 if ey > bby else -12), T_WALL + TIN_H + 14),
               (hole[0] - 8 + off, hole[1] + 6, T_WALL + 40),
               (hole[0] + off * 0.6, hole[1], Z_ROOF - 10), (hole[0] + off * 0.6, hole[1], Z_CAP + 18),
               (hole[0] - 25, hole[1] - 5 + k, Z_CAP + 25), (start[0], start[1], start[2] + 6), start]
        wires.append(wire(f"wire {nm}", pts, M[cm]))
    # screen cables: mini-HDMI and the screen's touch/power USB via the OTG port
    drv_top = Z_CAP + SCR_T + DRV_T
    wires.append(wire("HDMI cable", [(pzx - 20, pzy - 18, pzz + 2.4), (pzx - 26, pzy - 40, pzz + 18),
                                     (-20, scr_cy + 34, drv_top + 4), (-20, scr_cy + 22, drv_top)],
                      M["cable"], radius=2.2))
    wires.append(wire("Screen USB cable", [(pzx + 9, pzy - 18, pzz + 2.1), (pzx + 12, pzy - 36, pzz + 14),
                                           (20, scr_cy + 36, drv_top + 4), (20, scr_cy + 24, drv_top)],
                      M["cable"], radius=1.6))
    # power in: Pi PWR port -> out through the cable slot in the back wall -> down to the table
    slot_c = (X0 + SLOT_X + SLOT_W / 2, EXT_H - SLOT_TOP - SLOT_H / 2)
    wires.append(wire("USB power in", [(pzx + 22, pzy - 18, pzz + 2.1), (pzx + 22, pzy - 30, pzz + 10),
                                       (pzx - 10, pzy - 30, pzz + 26), (slot_c[0], IN_D - 12, slot_c[1]),
                                       (slot_c[0], EXT_D + 12, slot_c[1]), (slot_c[0] + 10, EXT_D + 60, 30),
                                       (slot_c[0] + 30, EXT_D + 140, 1.5), (slot_c[0] + 60, EXT_D + 260, 1.5)],
                      M["wht"], radius=1.8))
    part("wires", wires, (0, 0, 0))

    # Pi header -> INA226: supply in, logic power and I2C (all stay inside the cap)
    meter = []
    for k, (nm, cm, pin, dst) in enumerate([("5V in", "red", 2, "VIN+"), ("3V3", "org", 1, "VCC"),
                                            ("SDA", "wht", 3, "SDA"), ("SCL", "gry", 5, "SCL"),
                                            ("GND", "blk", 6, "GND")]):
        st, en = pi_pin(pin), ina_top[dst]
        meter.append(wire(f"meter wire {nm}", [st, (st[0], st[1], st[2] + 6),
                                               ((st[0] + en[0]) / 2, (st[1] + en[1]) / 2 - 2 * k, Z_CAP + 34 - 2 * k),
                                               (en[0], en[1], en[2] + 6), en], M[cm], radius=0.8))
    part("meter wires", meter, (0, 0, 0))

    # Pi pin 13 (GPIO27) and pin 14 (GND) -> the two button legs
    bwires = []
    for k, (nm, cm, pin) in enumerate([("GPIO27", "pur", 13), ("GND", "blk", 14)]):
        st, lg = pi_pin(pin), legs[k]
        bwires.append(wire(f"button wire {nm}", [st, (st[0], st[1], st[2] + 8),
                                                 (st[0] + 20, 140, Z_CAP + 38 - 3 * k),
                                                 (lg[0], 60, Z_CAP + 36 - 3 * k), (lg[0], lg[1] + 6, lg[2] + 2),
                                                 (lg[0], lg[1] + 3, lg[2])], M[cm], radius=0.8))
    part("button wires", bwires, (0, 0, 0))

    # ---------------------------------------------------------------- labels on the front
    ttl = text("Front title", "SILICON NEURON", (0, -0.6, Z_CAP + CAP_H / 2 + 5), 13, M["cyan"], extrude=0.4)
    sub = text("Front sub", "spiking vs artificial - measured live", (0, -0.6, Z_CAP + CAP_H / 2 - 10), 5.5,
               M["cyan"], extrude=0.2)
    part("cap", [ttl, sub], (0, 0, 250))

    # ---------------------------------------------------------------- table, world, lights
    table = box("Table", (1200, 900, 20), (0, 150, -10), M["table"], "Studio")
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.011, 0.014, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.world = world

    def light(name, kind, loc, energy, size=0.3, rot=(0, 0, 0), color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = color
        if kind == "AREA":
            ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = Vector(loc)
        lo.rotation_euler = Euler(rot)
        col("Studio").objects.link(lo)
        return lo

    light("Key", "AREA", (0.55, -0.5, 0.7), 18, 0.5, rot=(math.radians(50), 0, math.radians(45)),
          color=(1.0, 0.92, 0.85))
    light("Tin spot", "SPOT", (0.0, 0.10, 0.105), 0.8, rot=(math.radians(53), 0, 0), color=(1.0, 0.8, 0.6))
    light("Rim", "AREA", (-0.6, 0.6, 0.5), 10, 0.4, rot=(math.radians(-60), 0, math.radians(-135)),
          color=(0.7, 0.8, 1.0))

    # ---------------------------------------------------------------- optics diagram (section view only)
    diag = []
    ys = [ACT_Y0 + 12, (ACT_Y0 + ACT_Y1) / 2, ACT_Y1 - 12]
    eye = (0, -165)
    for yk in ys:
        zp = yk - PANE_Y0 + T_WALL                     # where the ray meets the pane (plane y - z = PANE_Y0 - T_WALL)
        # ray down from the screen to the pane
        diag.append(rod(f"ray screen {yk:.0f}", (IN_W / 2 + 12, yk, Z_ROOF), (IN_W / 2 + 12, yk, zp), M["ray_in"]))
        # reflected ray to the eye (horizontal)
        diag.append(rod(f"ray eye {yk:.0f}", (IN_W / 2 + 12, yk, zp), (IN_W / 2 + 12, eye[1], zp), M["ray_out"]))
        # virtual continuation behind the pane to the ghost plane
        diag.append(rod(f"ray virtual {yk:.0f}", (IN_W / 2 + 12, yk, zp), (IN_W / 2 + 12, GHOST_Y, zp),
                        M["ray_virt"], r=0.4))
    gz0, gz1 = T_WALL + (ACT_Y0 - PANE_Y0), T_WALL + (ACT_Y1 - PANE_Y0)
    diag.append(box("Ghost plane", (PANE_W, 0.6, gz1 - gz0), (0, GHOST_Y, (gz0 + gz1) / 2), M["ghost"], "Diagram"))
    diag.append(text("lbl ghost", "the image appears on this plane", (IN_W / 2 + 14, GHOST_Y + 4, gz1 - 6), 9,
                     M["ray_virt"], "Diagram", rot=(math.pi / 2, 0, math.pi / 2), align="LEFT"))
    diag.append(text("lbl screen", "screen, face down", (IN_W / 2 + 14, scr_cy, Z_CAP + 30), 9,
                     M["ray_in"], "Diagram", rot=(math.pi / 2, 0, math.pi / 2)))
    diag.append(text("lbl eye", "to the viewer's eye", (IN_W / 2 + 14, -110, ys[1] - PANE_Y0 + T_WALL + 10), 9,
                     M["ray_out"], "Diagram", rot=(math.pi / 2, 0, math.pi / 2)))
    diag.append(text("lbl pane", "clear acrylic at 45 deg", (IN_W / 2 + 14, PANE_Y0 + 5, 20), 9,
                     M["ray_virt"], "Diagram", rot=(math.pi / 2, 0, math.pi / 2)))
    diag.append(text("lbl tin", "Pico in the mint tin", (IN_W / 2 + 14, ty0 + 8, T_WALL + 96), 9,
                     M["ray_virt"], "Diagram", rot=(math.pi / 2, 0, math.pi / 2)))

    # ---------------------------------------------------------------- cameras
    def camera(name, loc, target, lens=50, ortho=None):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.clip_start, cd.clip_end = 0.005, 20
        if ortho:
            cd.type = "ORTHO"
            cd.ortho_scale = ortho
        co = bpy.data.objects.new(name, cd)
        co.location = Vector(loc) * MM
        d = Vector(target) * MM - co.location
        co.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        col("Studio").objects.link(co)
        return co

    cams = {
        "hero": camera("Cam hero", (300, -560, 165), (0, 120, 80), lens=55),
        "front": camera("Cam front (viewer eye)", (0, -620, 95), (0, 120, 80), lens=60),
        "section": camera("Cam section", (900, 90, 110), (0, 90, 110), ortho=0.52),
        "exploded": camera("Cam exploded", (380, -900, 640), (-20, 80, 150), lens=38),
    }

    labels = [("1 floor", "floor"), ("2 side walls", "side L"), ("3 back", "back"), ("4 roof + window", "roof"),
              ("5 acrylic pane 45deg", "pane"), ("7 7in screen, face down", "display"),
              ("8 cap + lid", "cap"), ("9 Pi Zero 2 W", "pi"), ("10 INA226", "ina226"), ("11 button", "button"),
              ("12 mint tin + Pico", "tin"), ("13 valance", "valance")]
    cam_rot = cams["exploded"].rotation_euler.copy()
    for txt, key in labels:
        objs, off = parts[key]
        c = sum((o.location for o in objs), Vector()) / len(objs) / MM + Vector(off)
        top = max(o.location.z + o.dimensions.z / 2 for o in objs) / MM + off[2]
        lift = {"side L": (0, -60, 0), "floor": (150, -40, -30), "pane": (0, -40, -20), "back": (150, 0, -70),
                "tin": (0, -40, 62), "button": (70, -20, -12), "ina226": (70, 0, -18)}.get(key, (0, 0, 0))
        pos = Vector((c.x, c.y, top + 16)) + Vector(lift)
        text("xl " + txt, txt, pos, 12, M["label"], "Exploded labels", rot=cam_rot)

    def set_exploded(on):
        for key, (objs, off) in parts.items():
            for ob in objs:
                base = ob.get("base_loc")
                if base is None:
                    ob["base_loc"] = list(ob.location)
                    base = ob["base_loc"]
                ob.location = Vector(base) + (Vector(off) * MM if on else Vector((0, 0, 0)))

    def show(names, visible):
        for n in names:
            c = cols.get(n)
            if c:
                c.hide_render = not visible
                c.hide_viewport = not visible

    # default viewport state: assembled, diagram hidden
    set_exploded(False)
    show(["Diagram", "Exploded labels"], False)
    scene.camera = cams["hero"]
    build.parts, build.cols, build.M, build.mat = parts, cols, M, mat
    build.parts = parts                     # read by assembly_animation.py
    build.cams = cams
    build.wire_idx = (WIRE_LOOSE, WIRE_ROOF)
    return scene, cams, set_exploded, show, bpy


def render_all(scene, cams, set_exploded, show, bpy, outdir, samples=48, res=(1400, 900), only=None):
    os.makedirs(outdir, exist_ok=True)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    try:
        scene.cycles.use_denoising = True
    except Exception:
        pass
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.film_transparent = False
    for vt in ("AgX", "Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = vt
            break
        except TypeError:
            continue
    jobs = [
        ("hero", False, [], ["Diagram", "Exploded labels"]),
        ("front", False, [], ["Diagram", "Exploded labels"]),
        ("section", False, ["Diagram"], ["Cap", "Exploded labels", "Wiring"]),
        ("exploded", True, ["Exploded labels"], ["Diagram", "Wiring", "Studio table"]),
    ]
    pane = bpy.data.objects.get("Acrylic pane 45deg")
    table = bpy.data.objects.get("Table")
    side_r = bpy.data.objects.get("Side R")
    for name, exploded, on, off in jobs:
        if only and name not in only:
            continue
        set_exploded(exploded)
        show(["Diagram", "Cap", "Wiring"], True)
        show(on, True)
        if table:
            table.hide_render = name == "exploded"
        bg = scene.world.node_tree.nodes["Background"].inputs[0]
        bg.default_value = (0.16, 0.17, 0.19, 1) if name == "exploded" else (0.01, 0.011, 0.014, 1)
        if pane:
            pane.data.materials[0] = bpy.data.materials["pane (exploded view tint)" if name == "exploded"
                                                        else "clear acrylic pane"]
        show(off, False)
        if side_r:
            side_r.hide_render = name == "section"      # cut away the near wall for the section
        scene.camera = cams[name]
        scene.render.filepath = os.path.join(outdir, f"{name}.png")
        bpy.ops.render.render(write_still=True)
        print("rendered", scene.render.filepath)
    set_exploded(False)
    show(["Cap", "Wiring"], True)
    show(["Diagram", "Exploded labels"], False)
    if table:
        table.hide_render = False
    if side_r:
        side_r.hide_render = False
    scene.camera = cams["hero"]


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    write_cut_sheet(os.path.join(HERE, "cut_sheet.svg"))
    scene, cams, set_exploded, show, bpy = build()
    blend = os.path.join(HERE, "silicon_neuron_demo.blend")
    try:
        bpy.ops.file.pack_all()            # embed the screen image so the .blend travels
    except Exception as e:
        print("could not pack images:", e)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    print("saved", blend)
    if "--render" in argv:
        samples = 48
        for a in argv:
            if a.startswith("--samples="):
                samples = int(a.split("=")[1])
        render_all(scene, cams, set_exploded, show, bpy, os.path.join(HERE, "renders"), samples=samples)
