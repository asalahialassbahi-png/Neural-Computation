"""
Step-by-step assembly animation — a LEGO-style building guide you can scrub.

Builds the model with build_demo.py, then keyframes every part flying into
place one step at a time, with a caption per step and timeline markers named
"Step 1 ..." so you can jump straight to any step in Blender.

    In Blender:   Scripting tab -> Open -> assembly_animation.py -> Run Script,
                  then press Space to play, or drag the timeline.
    Terminal:     blender -b -P assembly_animation.py -- --render
    pip bpy:      python assembly_animation.py --render [--every=2] [--res=1280x720]

--render writes PNG frames to renders/assembly/ and, if ffmpeg is installed,
renders/assembly.mp4. Rendering uses Workbench (flat, outlined, instruction-
manual look) because it is ~20x faster than EEVEE on a laptop CPU.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, HERE)
import build_demo as bd                    # noqa: E402

import bpy                                 # noqa: E402
from mathutils import Vector               # noqa: E402

MM = 0.001
FPS = 24
STEP = 96                                  # frames per step (4 s)
ARRIVE = (14, 62)                          # parts travel between these frames of a step
TIN_OFF = Vector((330, -130, 0))           # where the tin is built, on the table beside the box

scene, cams, set_exploded, show, _ = bd.build()
parts = bd.build.parts
set_exploded(False)
show(["Diagram"], False)
show(["Exploded labels"], True)               # each label's own keys hide it outside the kit step
for c in ("Cap", "Wiring", "Tin", "Display", "Electronics", "Enclosure", "Optics", "Studio"):
    show([c], True)


# ======================================================================= helpers
def obj(name):
    return bpy.data.objects[name]


def group(*keys, names=()):
    out = []
    for k in keys:
        out += parts[k][0]
    out += [obj(n) for n in names]
    return out


def rig(name, objs, parent=None):
    """An empty that carries a group of objects; animating it moves them all."""
    e = bpy.data.objects.new("rig " + name, None)
    e.empty_display_size = 0.01
    scene.collection.objects.link(e)
    if parent is not None:
        e.parent = parent
    for o in objs:                 # every rig sits at the origin, so local == world
        o.parent = e
    return e


def key_loc(o, frame, loc_mm):
    o.location = Vector(loc_mm) * MM
    o.keyframe_insert("location", frame=frame)


def key_vis(objs, frame, visible):
    for o in objs:
        o.hide_render = not visible
        o.hide_viewport = not visible
        o.keyframe_insert("hide_render", frame=frame)
        o.keyframe_insert("hide_viewport", frame=frame)


def all_objects_in(objs):
    out = []
    for o in objs:
        out.append(o)
        out += all_objects_in(o.children)
    return out


# ================================================================ split the groups
cap_front = [o for o in parts["cap"][0] if not o.name.startswith("Cap lid")]
lid = [o for o in parts["cap"][0] if o.name.startswith("Cap lid")]
wire_names = {w.name: w for w in parts["wires"][0]}
screen_cables = [wire_names.pop("HDMI cable"), wire_names.pop("Screen USB cable")]
usb = [wire_names.pop("USB power in")]
pico_wires = list(wire_names.values())          # drawn from the tin end (control point 0) to the Pi
meter_wires = parts["meter wires"][0]
button_wires = parts["button wires"][0]
WIRE_LOOSE, WIRE_ROOF = bd.build.wire_idx

# screen "off" cover under the lit area, removed at power-on
off = bpy.data.objects.new("Screen off", obj("Display lit area").data.copy())
off.data.materials.clear()
m_off = bpy.data.materials.new("screen off")
m_off.diffuse_color = (0.01, 0.01, 0.012, 1)
off.data.materials.append(m_off)
src = obj("Display lit area")
off.location = src.location.copy()
off.rotation_euler = src.rotation_euler.copy()
off.location.z -= 0.0003
bpy.data.collections["Display"].objects.link(off)
parts["display"][0].append(off)

# hologram illustration: the reflected image standing on the ghost plane
img_path = os.path.join(HERE, "screen_frame.png")
ghost = None
if os.path.exists(img_path):
    me = bpy.data.meshes.new("ghost")
    hw, z0, z1 = bd.ACT_W / 2 * MM, (bd.T_WALL + bd.ACT_Y0 - bd.PANE_Y0) * MM, (bd.T_WALL + bd.ACT_Y1 - bd.PANE_Y0) * MM
    y = bd.GHOST_Y * MM
    me.from_pydata([(-hw, y, z0), (hw, y, z0), (hw, y, z1), (-hw, y, z1)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for i, co in enumerate([(1, 0), (0, 0), (0, 1), (1, 1)]):   # u flipped: the mirror undoes hologram.py's flip
        uv.data[i].uv = co
    ghost = bpy.data.objects.new("Hologram (what the viewer sees)", me)
    mg = bpy.data.materials.new("hologram")
    mg.use_nodes = True
    nt = mg.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(img_path)
    bsdf = nt.nodes["Principled BSDF"]
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mg.diffuse_color = (0.3, 0.9, 1.0, 0.35)
    try:
        mg.surface_render_method = "BLENDED"
    except AttributeError:
        mg.blend_method = "BLEND"
    me.materials.append(mg)
    bpy.data.collections["Display"].objects.link(ghost)

# ================================================================ rigs
R = {}
R["tin asm"] = rig("tin assembly", [])
R["tin"] = rig("tin", group("tin"), R["tin asm"])
R["breadboard"] = rig("breadboard", group("breadboard"), R["tin asm"])
R["diode"] = rig("diode", group("diode"), R["tin asm"])
R["pico"] = rig("pico", group("pico"), R["tin asm"])
for k in ("floor", "side L", "side R", "back", "pane", "roof", "valance", "display", "pi", "ina226", "button"):
    R[k] = rig(k, group(k))
R["rails L"] = rig("rails L", group("rails"))
R["rails R"] = rig("rails R", group("rails R"))
R["cap front"] = rig("cap front", cap_front)
R["lid"] = rig("lid", lid)

WIRES = pico_wires + meter_wires + screen_cables + button_wires + usb
for w in WIRES:
    # SEGMENTS: factor k / (points - 1) ends exactly on control point k
    w.data.bevel_factor_mapping_start = w.data.bevel_factor_mapping_end = "SEGMENTS"


def seg_factor(w, k):
    return k / (len(w.data.splines[0].bezier_points) - 1)

# ================================================================ steps
# each: title, instruction lines, rig keys that arrive, approach offset (mm),
#       wires that grow [(wire, from, to)], camera (location, target) in mm.
# The list index IS the step number: 0 cover, 1 kit, 2-21 the build.
TINC = Vector((0, bd.GHOST_Y + 10 + 30.5, 12))
T = TINC + TIN_OFF


def full(ws):
    return [(w, 0.0, 1.0) for w in ws]


def stage(ws, k0, k1):
    """Pico jumpers: grow from control point k0 to k1 (None = the Pi end)."""
    return [(w, seg_factor(w, k0) if k0 else 0.0, 1.0 if k1 is None else seg_factor(w, k1)) for w in ws]


STEPS = [
    ("The finished demo", ["A spiking network and a normal one race on a Raspberry Pi Pico;",
                           "their spikes and measured energy float in mid-air."],
     [], None, [], ((300, -560, 165), (0, 120, 80))),
    ("What's in the kit", ["Every part, exploded. The number on each part is the step that uses it."],
     [], None, [], ((470, -1300, 980), (20, 40, 215))),
    ("Mint tin", ["Build the Pico sub-assembly on the table first.",
                  "Empty mint tin, open, lid hinge at the back."],
     ["tin"], (0, 0, 60), [], (T + Vector((150, -210, 180)), T)),
    ("Half-size breadboard", ["Peel the backing and stick the 400-point breadboard into the tin,",
                              "rows running left-right, red (+) rail at the BACK."],
     ["breadboard"], (0, 0, 60), [], (T + Vector((120, -170, 150)), T)),
    ("Raspberry Pi Pico", ["Flash the .uf2 first (hold BOOTSEL, plug in, drag the file), unplug USB.",
                           "Press it into rows 1-20, USB end LEFT, pins in columns c and h."],
     ["pico"], (0, 0, 60), [], (T + Vector((115, -170, 140)), T + Vector((-12, 0, 8)))),
    ("Schottky diode 1N5819", ["Plain end (anode) into the back red + rail at row 6; banded end",
                               "(cathode) into j2, the strip of Pico pin 39, VSYS."],
     ["diode"], (0, 0, 40), [], (T + Vector((-2, -78, 112)), T + Vector((-24, 8, 4)))),
    ("Floor", ["Black foam board, 196 x 220 mm, flat on the table."],
     ["floor"], None, [], ((420, -520, 420), (0, 110, 40))),
    ("Side walls", ["Hot-glue both 225 x 180 mm sides against the floor edges.",
                    "Check the corners with a set square."],
     ["side L", "side R"], None, [], ((420, -520, 420), (0, 110, 70))),
    ("Back wall", ["196 x 180 mm. The 24 x 12 mm cable slot goes at the TOP, on the LEFT",
                   "(seen from the front), next to where the Pi Zero will sit."],
     ["back"], None, [], ((380, -500, 380), (0, 130, 80))),
    ("Place the Pico tin", ["Before the pane: tin front edge 150 mm behind the opening, centred.",
                            "Plug the five jumpers into the + rail and a1-a4; loose ends to the back right."],
     ["tin asm"], None, stage(pico_wires, 0, WIRE_LOOSE), ((380, -460, 220), (0, 150, 30))),
    ("Pane rails", ["On each side wall draw a 45 degree line from the floor, 20 mm behind",
                    "the front, up to 125 mm. Glue two rails along it, 3 mm apart."],
     ["rails L", "rails R"], None, [], ((360, -380, 260), (0, 80, 60))),
    ("Acrylic pane", ["Slide the 190 x 170 mm, 2 mm sheet down the slots, film still on.",
                      "It touches the floor at the front and reaches 140 mm back at the top."],
     ["pane"], (0, -120, 160), [], ((360, -380, 260), (0, 80, 60))),
    ("Roof", ["Glue the roof 125 mm above the table, screen window towards the back.",
              "Feed the five loose jumpers up through the 14 mm hole at the back right."],
     ["roof"], None, stage(pico_wires, WIRE_LOOSE, WIRE_ROOF), ((-330, -420, 500), (10, 120, 105))),
    ("Valance", ["The 8 mm strip under the front edge of the roof hides the screen",
                 "from anyone crouching."],
     ["valance"], None, [], ((200, -380, 160), (0, 20, 110))),
    ("Screen, face down", ["Lay the 7-inch screen FACE DOWN over the window, its lit area's back",
                           "edge right above the top of the pane. Tape the bezel."],
     ["display"], (0, 0, 120), [], ((250, -180, 480), (0, 100, 130))),
    ("Pi Zero 2 W and INA226", ["Foam-tape both to the roof behind the screen:",
                                "Pi on the left (header at the back), meter on the right."],
     ["pi", "ina226"], (0, 0, 90), [], ((230, -40, 420), (0, 190, 135))),
    ("Meter wires", ["Pi pin 2 (5 V) to VIN+ (red),  pin 1 (3.3 V) to VCC (orange),",
                     "pin 3 to SDA (white),  pin 5 to SCL (grey),  pin 6 to GND (black)."],
     [], None, full(meter_wires), ((180, 60, 330), (0, 190, 140))),
    ("Screen cables", ["Mini-HDMI from the Pi to the screen; screen USB to the Pi's",
                       "middle (OTG) port with the adapter."],
     [], None, full(screen_cables), ((230, -80, 400), (-30, 150, 140))),
    ("Pico wires", ["Red from the + rail to meter VIN-;  a2 (yellow) to Pi pin 8,  a1 (green) to pin 10,",
                    "a4 (blue) to pin 11,  a3 (black) to pin 9.  (Right wall hidden here.)"],
     [], None, stage(pico_wires, WIRE_ROOF, None), ((600, 20, 300), (0, 172, 100))),
    ("Button and cap front", ["Fit the 24 mm button in the cap front, glue the cap front on, then",
                              "wire Pi pin 13 (purple) and pin 14 (black) to the button's legs."],
     ["button", "cap front"], None, full(button_wires), ((330, 470, 470), (10, 70, 140))),
    ("Lid and power", ["Lay the lid on (leave it unglued) and run the Pi's power lead",
                       "through the slot in the back wall to a 5 V supply."],
     ["lid"], (0, 0, 120), full(usb), ((-420, 640, 400), (-20, 180, 90))),
    ("Power on", ["Dim the lights. The display floats in front of the Pico that computes it.",
                  "Press the button for the next digit."],
     [], None, [], ((0, -620, 95), (0, 120, 80))),
]

# default approach directions: the exploded-view offsets, shortened
def approach(key, given):
    if given is not None:
        return Vector(given)
    if key in parts:
        return Vector(parts[key][1]) * 0.8
    if key in ("rails L", "rails R"):
        return Vector((0, -150, 80))
    if key == "tin asm":
        return TIN_OFF
    return Vector((0, 0, 80))


# final (assembled) location of each rig relative to its parent
HOME = {k: Vector((0, 0, 0)) for k in R}
HOME["tin asm"] = Vector((0, 0, 0))

# ================================================================ keyframing
scene.render.fps = FPS
scene.frame_start = 1
n_steps = len(STEPS)
scene.frame_end = n_steps * STEP
cut_away = obj("Side R")

every = [o for o in bpy.data.objects if o.type in ("MESH", "CURVE", "FONT")
         and o.users_collection and o.users_collection[0].name not in ("Diagram", "Exploded labels", "Studio")]
labels = list(bpy.data.collections["Exploded labels"].objects)
studio = [o for o in bpy.data.collections["Studio"].objects if o.type == "MESH"]

# kit labels: renumbered to the step that uses each part, plus the tin's contents
RELABEL = {"1 floor": "6 floor", "2 side walls": "7 side walls", "3 back": "8 back", "4 roof + window": "12 roof",
           "5 acrylic pane 45deg": "11 acrylic pane", "7 7in screen, face down": "14 7in screen",
           "8 cap + lid": "19 cap front, 20 lid", "9 Pi Zero 2 W": "15 Pi Zero 2 W", "10 INA226": "15 INA226",
           "11 button": "19 button", "12 mint tin + Pico": "2 mint tin", "13 valance": "13 valance"}
for o in bpy.data.collections["Exploded labels"].objects:
    o.data.body = RELABEL.get(o.data.body, o.data.body)
KIT_OFF = {"rails L": Vector((-75, 0, 0)), "rails R": Vector((75, 0, 0)),
           "breadboard": Vector((0, 0, 30)), "pico": Vector((0, 0, 58)), "diode": Vector((0, 0, 80))}
lab_src = next(iter(bpy.data.collections["Exploded labels"].objects))
bpy.context.view_layer.update()
for txt, key, at in (("3 breadboard", "breadboard", (-95, -20, 0)), ("4 Pico", "pico", (-80, -20, 0)),
                     ("5 diode", "diode", (60, 0, 8)), ("10 rails", "rails R", (0, -30, 60))):
    lo = lab_src.copy()
    lo.data = lab_src.data.copy()
    lo.data.body = txt
    ctr = sum((o.matrix_world.translation for o in all_objects_in([R[key]]) if o.type != "EMPTY"),
              Vector()) / len([o for o in all_objects_in([R[key]]) if o.type != "EMPTY"]) / MM
    base = ctr + KIT_OFF[key] + (TIN_OFF if key in ("breadboard", "pico", "diode") else Vector())
    lo.location = (base + Vector(at)) * MM
    lo.name = "xl " + txt
    bpy.data.collections["Exploded labels"].objects.link(lo)
labels = list(bpy.data.collections["Exploded labels"].objects)
m_lab = bpy.data.materials.new("kit label")
m_lab.diffuse_color = (0.02, 0.02, 0.025, 1)                  # dark text on the light background
for o in labels:
    o.data.materials[0] = m_lab
    o.scale = (1.5, 1.5, 1.5)

# step 0 (cover): everything assembled and powered
f1 = 1
key_vis(every, f1, True)
key_vis(labels, f1, False)
for k, r in R.items():
    key_loc(r, f1, HOME[k])
for w in WIRES:
    w.data.bevel_factor_start = 0.0
    w.data.bevel_factor_end = 1.0
    w.data.keyframe_insert("bevel_factor_end", frame=f1)
key_vis([off], f1, False)
if ghost:
    key_vis([ghost], f1, True)

# step 1 (kit): exploded, labelled; wires are shown as they are fitted
f2 = STEP + 1
key_vis([off], f2, True)
key_vis(WIRES, f2, False)
if ghost:
    key_vis([ghost], f2, False)
key_vis(labels, f2, True)
for k, r in R.items():
    key_loc(r, f2 - 1, HOME[k])
    off_mm = TIN_OFF if k == "tin asm" else (Vector(parts[k][1]) if k in parts and k not in
                                              ("tin", "breadboard", "diode", "pico") else Vector((0, 0, 0)))
    if k in KIT_OFF:
        off_mm = KIT_OFF[k]
    if k == "cap front":
        off_mm = Vector(parts["cap"][1])
    if k == "lid":
        off_mm = Vector(parts["cap"][1]) + Vector((0, 0, 60))
    key_loc(r, f2 + 10, off_mm)
    key_loc(r, 2 * STEP, off_mm)

# from step 3 on: hide everything, then reveal part by part
f3 = 2 * STEP + 1
key_vis(labels, f3, False)
key_vis(every, f3, False)
key_vis(studio, f3, True)
for w in WIRES:
    w.data.bevel_factor_end = 0.0
    w.data.keyframe_insert("bevel_factor_end", frame=f3)
key_vis(WIRES, f3, False)                        # hidden until they start to grow
for k, r in R.items():
    if k == "tin asm":
        key_loc(r, f3, TIN_OFF)
    else:
        key_loc(r, f3, HOME[k])

for i, (title, lines, keys, appr, grow, cam) in enumerate(STEPS):
    if i < 2:
        continue
    s0 = i * STEP + 1
    a0, a1 = s0 + ARRIVE[0], s0 + ARRIVE[1]
    for k in keys:
        r = R[k]
        objs = [o for o in all_objects_in([r]) if o.type != "EMPTY"]
        if k == "tin asm":
            key_loc(r, a0, TIN_OFF)
            key_loc(r, a1, HOME[k])
            continue
        key_vis([o for o in objs if o not in WIRES], a0 - 6, True)
        start = HOME[k] + approach(k, appr)
        key_loc(r, s0, start)
        key_loc(r, a0, start)
        key_loc(r, a1, HOME[k])
    # wires grow from the end already fixed; after any part in this step has landed
    gap, dur, first = (4, 18, a1 - 4) if keys else (6, 30, a0)
    for j, (w, fa, fb) in enumerate(grow):
        g0 = first + j * gap
        g1 = min(g0 + dur, s0 + STEP - 8)
        if fa == 0.0:
            key_vis([w], g0, True)
        w.data.bevel_factor_end = fa
        w.data.keyframe_insert("bevel_factor_end", frame=g0)
        w.data.bevel_factor_end = fb
        w.data.keyframe_insert("bevel_factor_end", frame=g1)

for i, (title, *_r) in enumerate(STEPS):
    scene.timeline_markers.new(f"Step {i}: {title}", frame=i * STEP + 1)

# right wall as a cut-away while the tin goes in and the Pico wires are routed
i_tin = [s[0] for s in STEPS].index("Place the Pico tin")
i_pw = [s[0] for s in STEPS].index("Pico wires")
key_vis([cut_away], i_tin * STEP + 1, False)
key_vis([cut_away], (i_tin + 1) * STEP + 1, True)
key_vis([cut_away], i_pw * STEP + 1, False)
key_vis([cut_away], (i_pw + 1) * STEP + 1, True)

# power on in the last step
last = (n_steps - 1) * STEP + 1
key_vis([off], last + 30, True)
key_vis([off], last + 31, False)
if ghost:
    key_vis([ghost], last + 30, False)
    key_vis([ghost], last + 36, True)

# ================================================================ camera
cam_d = bpy.data.cameras.new("Cam assembly")
cam_d.lens = 45
cam_d.clip_start, cam_d.clip_end = 0.005, 20
cam = bpy.data.objects.new("Cam assembly", cam_d)
scene.collection.objects.link(cam)
target = bpy.data.objects.new("cam target", None)
scene.collection.objects.link(target)
tc = cam.constraints.new("TRACK_TO")
tc.target, tc.track_axis, tc.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
scene.camera = cam
for i, st in enumerate(STEPS):
    loc, tgt = st[5]
    s0 = i * STEP + 1
    move_end = s0 + 16
    if i == 0:
        key_loc(cam, s0, loc)
        key_loc(target, s0, tgt)
    key_loc(cam, move_end, loc)
    key_loc(target, move_end, tgt)
    key_loc(cam, s0 + STEP - 1, loc)
    key_loc(target, s0 + STEP - 1, tgt)
# slow push-in on the final step
key_loc(cam, n_steps * STEP, (60, -480, 105))

# ================================================================ captions (parented to the camera)
M_cap = bpy.data.materials.new("caption")
M_cap.diffuse_color = (1, 1, 1, 1)
M_num = bpy.data.materials.new("step number")
M_num.diffuse_color = (1.0, 0.75, 0.1, 1)
M_band = bpy.data.materials.new("caption band")
M_band.diffuse_color = (0.05, 0.06, 0.08, 0.82)
try:
    M_band.surface_render_method = "BLENDED"
except AttributeError:
    M_band.blend_method = "BLEND"

D = 0.30                                        # caption plane distance from the camera (m)
half_w = D * 18 / cam_d.lens                    # 36 mm sensor width
half_h = half_w * 9 / 16


def cap_text(name, body, x, y, size, mat, align="LEFT"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "TOP"
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    ob.parent = cam
    ob.location = (x, y, -D)
    scene.collection.objects.link(ob)
    return ob


band_me = bpy.data.meshes.new("band")
bh = half_h * 0.34
band_me.from_pydata([(-half_w, half_h - bh, -D - 0.001), (half_w, half_h - bh, -D - 0.001),
                     (half_w, half_h, -D - 0.001), (-half_w, half_h, -D - 0.001)], [], [(0, 1, 2, 3)])
band = bpy.data.objects.new("caption band", band_me)
band.data.materials.append(M_band)
band.parent = cam
band.hide_render = True
scene.collection.objects.link(band)

x0 = -half_w * 0.95
for i, (title, lines, *_rest) in enumerate(STEPS):
    s0 = i * STEP + 1
    head = title if i == 0 else f"Step {i} of {n_steps - 1}  |  {title}"
    t = cap_text(f"caption {i} title", head, x0, half_h * 0.93, half_h * 0.075, M_num)
    b = cap_text(f"caption {i} body", "\n".join(lines), x0, half_h * 0.78, half_h * 0.052, M_cap)
    for o in (t, b):                        # viewport only: renders get captions from overlay_captions()
        o.hide_render = True
        o.hide_viewport = True
        o.keyframe_insert("hide_viewport", frame=1)
        o.hide_viewport = False
        o.keyframe_insert("hide_viewport", frame=s0)
        o.hide_viewport = True
        o.keyframe_insert("hide_viewport", frame=s0 + STEP)

# fcurves for visibility must be CONSTANT, locations eased
for o in bpy.data.objects:
    ad = o.animation_data
    if not ad or not ad.action:
        continue
    fcs = getattr(ad.action, "fcurves", None)
    if fcs is None:                                         # Blender 5 layered actions
        fcs = [fc for layer in ad.action.layers for strip in layer.strips
               for cb in strip.channelbags for fc in cb.fcurves]
    for fc in fcs:
        for kp in fc.keyframe_points:
            if fc.data_path.startswith("hide"):
                kp.interpolation = "CONSTANT"
            else:
                kp.interpolation = "BEZIER"
                kp.easing = "AUTO"

# ================================================================ look: instruction-manual Workbench
scene.render.engine = "BLENDER_WORKBENCH"
sh = scene.display.shading
sh.light = "STUDIO"
sh.color_type = "MATERIAL"
sh.show_object_outline = True
sh.object_outline_color = (0.05, 0.05, 0.05)
sh.show_cavity = True
sh.cavity_type = "WORLD"
scene.world.color = (0.80, 0.82, 0.86)
for m in bpy.data.materials:                                # workbench reads the viewport colour
    if m.use_nodes and m.node_tree and "Principled BSDF" in m.node_tree.nodes and m.name not in (
            "caption", "step number", "caption band", "screen off", "hologram", "kit label"):
        b = m.node_tree.nodes["Principled BSDF"]
        col = list(b.inputs["Base Color"].default_value)
        es = b.inputs.get("Emission Strength")
        ec = b.inputs.get("Emission Color") or b.inputs.get("Emission")
        if es is not None and es.default_value > 0.5 and ec is not None:
            col = list(ec.default_value)
        a = b.inputs["Alpha"].default_value if "Alpha" in b.inputs else 1.0
        m.diffuse_color = (col[0], col[1], col[2], a)
bpy.data.materials["foam board (matt black)"].diffuse_color = (0.2, 0.2, 0.215, 1)   # black board drawn mid-grey
m_rail = bpy.data.materials["foam board (matt black)"].copy()                         # rails a shade lighter
m_rail.name = "foam board strips (drawn lighter)"
m_rail.diffuse_color = (0.42, 0.42, 0.44, 1)
for o in group("rails", "rails R"):
    o.data.materials[0] = m_rail
bpy.data.materials["clear acrylic pane"].diffuse_color = (0.65, 0.85, 1.0, 0.22)
try:
    bpy.data.materials["clear acrylic pane"].surface_render_method = "BLENDED"
except AttributeError:
    bpy.data.materials["clear acrylic pane"].blend_method = "BLEND"
bpy.data.materials["screen image"].diffuse_color = (0.02, 0.02, 0.03, 1)
bpy.data.materials["table"].diffuse_color = (0.72, 0.62, 0.5, 1)
bpy.data.materials["tin plate"].diffuse_color = (0.72, 0.74, 0.77, 1)
for m in bpy.data.materials:                                # workbench draws metallic as near-black
    m.metallic = 0.0
    m.roughness = 0.5

scene.frame_set(1)


def write_steps_json(folder, every_n=1):
    import json
    info = {"fps": FPS // every_n, "every": every_n, "step_frames": STEP, "steps": [
        {"title": t, "lines": l, "start": i * STEP + 1, "end": (i + 1) * STEP} for i, (t, l, *_r) in enumerate(STEPS)]}
    with open(os.path.join(folder, "steps.json"), "w") as f:
        json.dump(info, f, indent=1)


def export_ghost_quads():
    """Screen position of the hologram plane in every frame where it is visible, in
    1280 x 720 pixels (make_video.py scales to the real size); make_video.py warps
    the real display image into it (Workbench cannot)."""
    import json
    quads = {}
    for f in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(f)
        q = ghost_quad((1280, 720))
        if q:
            quads[f] = q
    os.makedirs(os.path.join(HERE, "renders"), exist_ok=True)
    with open(os.path.join(HERE, "renders", "ghost_quads.json"), "w") as fh:
        json.dump(quads, fh)


def ghost_quad(res):
    """Screen-space corners of the hologram plane at the current frame, or None."""
    from bpy_extras.object_utils import world_to_camera_view
    if ghost is None or ghost.hide_render:
        return None
    W, H = res
    return [[c.x * W, (1 - c.y) * H] for c in
            (world_to_camera_view(scene, scene.camera, ghost.matrix_world @ v.co) for v in ghost.data.vertices)]


def preview(res=(640, 360), offsets=(89,)):
    """One test frame per step (the booklet frame, start + 89) with captions, plus a
    contact sheet, to check framing before the full render. --preview=40,89 adds frames."""
    import json
    import make_video
    from PIL import Image
    out = os.path.join(HERE, "renders", "preview")
    os.makedirs(out, exist_ok=True)
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.image_settings.file_format = "PNG"
    write_steps_json(out)
    info = json.load(open(os.path.join(out, "steps.json")))
    gimg = make_video.ghost_image()
    sheet = []
    for i in range(n_steps):
        for o in offsets:
            f = i * STEP + 1 + o
            scene.frame_set(f)
            path = os.path.join(out, f"raw{f:05d}.png")
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            im = Image.open(path).convert("RGB")
            q = ghost_quad(res)
            if q and gimg is not None:
                im = make_video.add_ghost(im, q, gimg)
            im = make_video.caption(im, i, info)
            im.save(os.path.join(out, f"f{f:05d}.png"))
            os.remove(path)
            sheet.append((f"{i}  (frame {f})", im))
    make_video.contact_sheet(sheet, os.path.join(HERE, "renders", "preview_sheet.jpg"))


def render(every_n=1, res=(1280, 720)):
    out = os.path.join(HERE, "renders", "assembly")
    os.makedirs(out, exist_ok=True)
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.image_settings.file_format = "PNG"
    frames = range(scene.frame_start, scene.frame_end + 1, every_n)
    for f in frames:
        path = os.path.join(out, f"f{f:05d}.png")
        if os.path.exists(path):
            continue
        scene.frame_set(f)
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
    write_steps_json(out, every_n)
    export_ghost_quads()
    import make_video                       # captions, MP4 and the step booklet
    make_video.main(out)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    blend = os.path.join(HERE, "silicon_neuron_assembly.blend")
    try:
        bpy.ops.file.pack_all()
    except Exception as e:
        print("could not pack images:", e)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    print("saved", blend, "-", n_steps, "steps,", scene.frame_end, "frames")
    pv = next((a for a in argv if a.startswith("--preview")), None)
    if pv:
        preview(offsets=tuple(int(v) for v in pv.split("=")[1].split(",")) if "=" in pv else (89,))
    if "--render" in argv:
        ev = int(next((a.split("=")[1] for a in argv if a.startswith("--every=")), 1))
        rs = next((a.split("=")[1] for a in argv if a.startswith("--res=")), "1280x720")
        render(ev, tuple(int(v) for v in rs.split("x")))
