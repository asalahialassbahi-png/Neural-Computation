"""
Captions, MP4 and a LEGO-style step booklet from the rendered assembly frames.

    python make_video.py                  # after assembly_animation.py --render
    python make_video.py renders/assembly

Writes
    renders/assembly.mp4           24 fps, a keyframe every second (smooth scrubbing),
                                   one chapter per step (VLC / QuickTime: jump by chapter)
    renders/assembly_booklet.pdf   one landscape page per step, like a LEGO manual
Needs Pillow and ffmpeg.
"""
import json
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "C:/Windows/Fonts"]


def font(name, size):
    for d in FONT_DIRS:
        for n in (name, "DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def caption(img, step_i, info):
    steps = info["steps"]
    st = steps[step_i]
    W, H = img.size
    k = W / 1280
    last = len(steps) - 1                         # step 0 is the cover, 1 the kit, 2.. the build
    d = ImageDraw.Draw(img, "RGBA")
    f_num = font("DejaVuSans-Bold.ttf", int(54 * k))
    f_title = font("DejaVuSans-Bold.ttf", int(28 * k))
    f_body = font("DejaVuSans.ttf", int(19 * k))
    f_small = font("DejaVuSans.ttf", int(14 * k))

    x0, y0, pad = int(24 * k), int(22 * k), int(16 * k)
    body_w = max([d.textlength(l, font=f_body) for l in st["lines"]] + [d.textlength(st["title"], font=f_title)])
    num_w = int(86 * k) if step_i >= 1 else 0
    panel_w = int(num_w + body_w + 3 * pad)
    panel_h = int(pad * 2 + 36 * k + len(st["lines"]) * 26 * k)
    d.rounded_rectangle([x0, y0, x0 + panel_w, y0 + panel_h], radius=int(14 * k), fill=(255, 255, 255, 235),
                        outline=(30, 30, 30, 255), width=max(1, int(2 * k)))
    tx = x0 + pad
    if step_i >= 1:
        d.rounded_rectangle([x0 + pad, y0 + pad, x0 + pad + int(70 * k), y0 + pad + int(70 * k)],
                            radius=int(10 * k), fill=(255, 196, 0, 255), outline=(30, 30, 30, 255),
                            width=max(1, int(2 * k)))
        num = str(step_i)
        nw = d.textlength(num, font=f_num)
        d.text((x0 + pad + int(35 * k) - nw / 2, y0 + pad + int(4 * k)), num, font=f_num, fill=(20, 20, 20))
        tx = x0 + pad + num_w
    d.text((tx, y0 + pad - int(2 * k)), st["title"], font=f_title, fill=(20, 20, 20))
    for j, line in enumerate(st["lines"]):
        d.text((tx, y0 + pad + int(38 * k) + j * int(26 * k)), line, font=f_body, fill=(40, 40, 40))

    # progress strip: one dot per step 1..last
    if step_i >= 1:
        by = H - int(28 * k)
        bx0, bx1 = int(120 * k), W - int(120 * k)
        d.line([bx0, by, bx1, by], fill=(40, 40, 40, 160), width=max(1, int(3 * k)))
        for s in range(1, last + 1):
            cx = bx0 + (bx1 - bx0) * (s - 1) / max(1, last - 1)
            r = int((9 if s == step_i else 5) * k)
            col = (255, 196, 0, 255) if s == step_i else ((40, 40, 40, 255) if s < step_i else (255, 255, 255, 255))
            d.ellipse([cx - r, by - r, cx + r, by + r], fill=col, outline=(30, 30, 30, 255))
        d.text((bx0 - int(100 * k), by - int(9 * k)), f"{step_i} / {last}", font=f_small, fill=(20, 20, 20))
    d.text((W - int(250 * k), H - int(60 * k) if step_i < 1 else int(8 * k)), "Silicon Neuron  |  assembly",
           font=f_small, fill=(60, 60, 60))
    return img


def _perspective_coeffs(dst, src):
    """Coefficients for PIL's PERSPECTIVE transform mapping output pixel dst[i] -> input src[i]."""
    import numpy as np
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()


def add_ghost(img, quad, ghost):
    """Screen-blend the hologram image into the projected ghost plane, as the eye would see it:
    black pixels vanish, bright ones glow on top of whatever is behind."""
    from PIL import ImageChops
    w, h = ghost.size
    src = [(0, h), (w, h), (w, 0), (0, 0)]            # quad order: bottom-left, bottom-right, top-right, top-left
    warped = ghost.transform(img.size, Image.PERSPECTIVE, _perspective_coeffs(quad, src), Image.BILINEAR)
    return ImageChops.screen(img, warped.point(lambda p: int(p * 0.9)))


def ghost_image():
    """The real display frame as the viewer sees it (the mirror un-flips hologram.py's flip)."""
    gpath = os.path.join(HERE, "screen_frame.png")
    if not os.path.exists(gpath):
        return None
    return Image.open(gpath).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)


def contact_sheet(items, path, cols=4, w=480):
    """items: [(label, PIL image)] -> one JPEG grid for checking every step at a glance."""
    h = int(w * items[0][1].size[1] / items[0][1].size[0])
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 22)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = font("DejaVuSans-Bold.ttf", 15)
    for n, (label, im) in enumerate(items):
        x, y = (n % cols) * w, (n // cols) * (h + 22)
        sheet.paste(im.resize((w, h)), (x, y + 22))
        d.text((x + 4, y + 3), label, font=f, fill=(0, 0, 0))
    sheet.save(path, quality=90)
    print("wrote", path)


def main(folder=None):
    folder = folder or os.path.join(HERE, "renders", "assembly")
    info = json.load(open(os.path.join(folder, "steps.json")))
    out = folder + "_captioned"
    os.makedirs(out, exist_ok=True)
    frames = sorted(f for f in os.listdir(folder) if f.startswith("f") and f.endswith(".png"))
    per = info["step_frames"]
    quads, ghost = {}, ghost_image()
    qpath = os.path.join(os.path.dirname(folder), "ghost_quads.json")
    if os.path.exists(qpath):
        quads = {int(k): v for k, v in json.load(open(qpath)).items()}
    for name in frames:
        dst = os.path.join(out, name)
        if os.path.exists(dst):
            continue
        fnum = int(name[1:6])
        i = min((fnum - 1) // per, len(info["steps"]) - 1)
        im = Image.open(os.path.join(folder, name)).convert("RGB")
        if fnum in quads and ghost is not None:
            k = im.size[0] / 1280
            im = add_ghost(im, [(x * k, y * k) for x, y in quads[fnum]], ghost)
        caption(im, i, info).save(dst)

    renders = os.path.dirname(folder)
    # ---- MP4 with chapters
    ff = shutil.which("ffmpeg")
    if ff:
        fps = info["fps"]
        meta = os.path.join(out, "chapters.txt")
        with open(meta, "w") as f:
            f.write(";FFMETADATA1\n")
            for i, st in enumerate(info["steps"]):
                t0 = int((st["start"] - 1) / info["every"] / fps * 1000)
                t1 = int(st["end"] / info["every"] / fps * 1000)
                title = f"Step {i}: {st['title']}"
                f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={t0}\nEND={t1}\ntitle={title}\n")
        mp4 = os.path.join(renders, "assembly.mp4")
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-pattern_type", "glob",
                        "-i", os.path.join(out, "f*.png"), "-i", meta, "-map_metadata", "1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-g", str(fps),
                        "-movflags", "+faststart", mp4], check=True)
        print("wrote", mp4)

    # ---- booklet: the settled frame near the end of every step, one per page
    pages = []
    for i, st in enumerate(info["steps"]):
        want = st["end"] - 6
        cands = [f for f in frames if int(f[1:6]) <= want]
        if not cands:
            continue
        pages.append(Image.open(os.path.join(out, cands[-1])).convert("RGB"))
    if pages:
        pdf = os.path.join(renders, "assembly_booklet.pdf")
        pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150)
        print("wrote", pdf, len(pages), "pages")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
