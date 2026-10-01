"""
Cycle audit: run the REAL compiled Pico inference code in an ARM emulator and
count its CPU cycles on real MNIST images.

    python cycle_audit.py              # 40 images per model
    python cycle_audit.py --n 200

What it does
  1. Compiles firmware/nn_core.c + firmware/generated/model_data.c with the
     same compiler flags as the firmware (arm-none-eabi-gcc -O2,
     -mcpu=cortex-m0plus) into a bare-metal ELF.
  2. Loads it into Unicorn (a CPU emulator) and calls ann_infer / snn_infer /
     snn_fast_infer exactly as main.c does.
  3. Counts cycles per executed basic block using the Cortex-M0+ instruction
     timings (ARM Cortex-M0+ Technical Reference Manual, table 3-1): 1 cycle
     for data processing, 2 for a load/store, 1+N for LDM/STM, 2 for a taken
     branch, 1 for a not-taken one, 3 for BL, and 1 for MULS because the
     RP2040 is built with the single-cycle multiplier.
  4. Fits the per-component costs in energy_model.CYC to the measured totals
     and reports how well the hand model predicts the measurement.

Not modelled: flash wait states. Weights live in flash behind a 16 kB XIP
cache; every model reads one weight row per processed input, so misses scale
with the same row count the model already uses (MATHS.md 8.3). The INA226 on
the real board measures all of it.

Writes results/cycle_audit.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

import config as C
from mnist_idx import load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FW = os.path.join(ROOT, "firmware")
FLASH, RAM, RAM_SIZE = 0x10000000, 0x20000000, 0x80000

LINKER = """
MEMORY { FLASH (rx) : ORIGIN = 0x10000000, LENGTH = 4M
         RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 512K }
SECTIONS {
  .text : { *(.text*) *(.rodata*) } > FLASH
  .data : { *(.data*) } > RAM
  .bss  : { *(.bss*) *(COMMON) } > RAM
}
"""
STUB = "void stop_here(void) { __asm__ volatile (\"bkpt #0\"); }\n"


def build(tmp):
    ld = os.path.join(tmp, "link.ld")
    open(ld, "w").write(LINKER)
    stub = os.path.join(tmp, "stub.c")
    open(stub, "w").write(STUB)
    elf = os.path.join(tmp, "core.elf")
    cmd = ["arm-none-eabi-gcc", "-mcpu=cortex-m0plus", "-mthumb", "-O2", "-std=c11",
           "-I" + FW, "-I" + os.path.join(FW, "generated"), "-nostartfiles", "-specs=nano.specs",
           "-Wl,-T," + ld, "-Wl,--gc-sections", "-Wl,-e,stop_here", "-Wl,--undefined=ann_infer",
           "-Wl,--undefined=snn_infer", "-Wl,--undefined=snn_fast_infer", "-o", elf,
           os.path.join(FW, "nn_core.c"), os.path.join(FW, "generated", "model_data.c"), stub]
    subprocess.run(cmd, check=True)
    return elf


class Emu:
    def __init__(self, elf):
        from capstone import Cs, CS_ARCH_ARM, CS_MODE_THUMB, CS_MODE_MCLASS
        from elftools.elf.elffile import ELFFile
        from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB, UC_MODE_MCLASS, UC_HOOK_BLOCK
        from unicorn.arm_const import UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_R0
        self.R0, self.SP, self.LR = UC_ARM_REG_R0, UC_ARM_REG_SP, UC_ARM_REG_LR
        self.uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
        self.uc.mem_map(FLASH, 4 << 20)
        self.uc.mem_map(RAM, RAM_SIZE)
        self.sym = {}
        with open(elf, "rb") as f:
            e = ELFFile(f)
            for seg in e.iter_segments():
                if seg["p_type"] == "PT_LOAD" and seg["p_filesz"]:
                    self.uc.mem_write(seg["p_vaddr"], seg.data())
            for s in e.get_section_by_name(".symtab").iter_symbols():
                if s.name:
                    self.sym[s.name] = s["st_value"]
        self.md = Cs(CS_ARCH_ARM, CS_MODE_THUMB | CS_MODE_MCLASS)
        self.blocks = {}                     # addr -> (cycles if fall-through, extra if taken, end addr)
        self.prev = None
        self.cycles = 0
        self.uc.hook_add(UC_HOOK_BLOCK, self._on_block)

    def _cost(self, addr, size):
        code = bytes(self.uc.mem_read(addr, size))
        cyc, taken_extra = 0, 0
        insns = list(self.md.disasm(code, addr))
        for k, i in enumerate(insns):
            m = i.mnemonic.split(".")[0]
            last = k == len(insns) - 1
            if m in ("ldm", "ldmia", "stm", "stmia", "push"):
                cyc += 1 + i.op_str.count(",") + 1
            elif m == "pop":
                n = i.op_str.count(",") + 1
                cyc += (3 + n) if "pc" in i.op_str else (1 + n)
            elif m.startswith("ldr") or m.startswith("str"):
                cyc += 2
            elif m == "bl":
                cyc += 3
            elif m in ("bx", "blx"):
                cyc += 2
            elif m == "b":
                cyc += 2
            elif m.startswith("b") and m not in ("bic", "bics", "bkpt") and last:
                cyc += 1                     # conditional branch, not taken ...
                taken_extra = 1              # ... +1 when taken
            elif (m in ("add", "mov")) and i.op_str.startswith("pc"):
                cyc += 2
            else:
                cyc += 1                     # data processing, MULS (single-cycle on RP2040)
        return cyc, taken_extra, addr + size

    def _on_block(self, uc, addr, size, _):
        if self.prev is not None:
            cyc, extra, end = self.prev
            self.cycles += cyc + (extra if addr != end else 0)
        b = self.blocks.get(addr)
        if b is None:
            b = self.blocks[addr] = self._cost(addr, size)
        self.prev = b

    def call(self, name, *args):
        uc = self.uc
        from unicorn.arm_const import UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3
        regs = [self.R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3]
        sp = RAM + RAM_SIZE - 64
        stack_args = list(args[4:])
        sp -= 4 * len(stack_args)
        for k, v in enumerate(stack_args):
            uc.mem_write(sp + 4 * k, int(v & 0xFFFFFFFF).to_bytes(4, "little"))
        for r, v in zip(regs, args[:4]):
            uc.reg_write(r, int(v) & 0xFFFFFFFF)
        uc.reg_write(self.SP, sp)
        stop = self.sym["stop_here"] & ~1
        uc.reg_write(self.LR, stop | 1)
        self.cycles, self.prev = 0, None
        uc.emu_start(self.sym[name] | 1, stop)
        if self.prev is not None:
            self.cycles += self.prev[0]
        r0 = uc.reg_read(self.R0)
        return r0 - (1 << 32) if r0 & 0x80000000 else r0, self.cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-poisson", type=int, default=20)
    a = ap.parse_args()
    if not shutil.which("arm-none-eabi-gcc"):
        raise SystemExit("needs arm-none-eabi-gcc (apt install gcc-arm-none-eabi libnewlib-arm-none-eabi)")
    tmp = tempfile.mkdtemp()
    emu = Emu(build(tmp))
    img0 = emu.sym["test_images"]
    scratch = RAM + RAM_SIZE - 0x4000                 # out-parameters live here
    has_fast = "snn_fast_infer" in emu.sym and "fast_W1" in emu.sym
    fast_margin = None
    if has_fast:                                     # the deployed exit margin, as main.c uses it
        for ln in open(os.path.join(FW, "generated", "model_data.h")):
            if ln.startswith("#define FAST_EXIT_MARGIN"):
                fast_margin = int(ln.split()[2])
    _, _, xte, yte = load_mnist()
    x = xte[:C.N_TEST_PICO]

    # reference predictions from the Python integer model: the emulated code must agree
    from intsim import quantise_ann, ann_int, quantise_snn, snn_int, input_spikes_int
    from export import WDIR
    xtr = load_mnist()[0]
    ref_a = ann_int(quantise_ann(dict(np.load(os.path.join(WDIR, "ann_float.npz"))), xtr[:5000]), x[:a.n])[0]
    ref_l = snn_int(quantise_snn(dict(np.load(os.path.join(WDIR, "snn_latency_float.npz")))),
                    input_spikes_int(x[:a.n], "latency", np.arange(a.n)))[0]
    mism = 0
    rows = {"A": [], "Z": [], "L": [], "P": [], "E": []}
    for idx in range(a.n):
        xp = img0 + idx * C.N_IN
        pa, ca = emu.call("ann_infer", xp, 0, 0)
        pz, cz = emu.call("ann_infer", xp, 1, 0)
        pl, cl = emu.call("snn_infer", xp, 0, idx, scratch, 0, 0)
        rows["A"].append(ca); rows["Z"].append(cz); rows["L"].append(cl)
        mism += int(pa != ref_a[idx]) + int(pz != ref_a[idx]) + int(pl != ref_l[idx])
        if idx < a.n_poisson:                        # rate code: slow to emulate, fewer images
            _, cp = emu.call("snn_infer", xp, 1, idx, scratch, 0, 0)
            rows["P"].append(cp)
        if has_fast:
            pe, ce = emu.call("snn_fast_infer", xp, fast_margin, scratch, scratch + 4, 0)
            rows["E"].append(ce)
        if idx % 10 == 0:
            print(f"image {idx}: A {ca / 1e3:.0f}k  Z {cz / 1e3:.0f}k  L {cl / 1e3:.0f}k cycles"
                  + (f"  E {rows['E'][-1] / 1e3:.0f}k" if has_fast else ""), flush=True)
    print(f"emulated predictions vs Python integer model: {mism} mismatches")
    out = {"n": a.n, "f_clk_hz": 125e6, "prediction_mismatches": mism, "models": {}}
    for k, v in rows.items():
        if v:
            v = np.array(v, float)
            out["models"][k] = {"cycles_mean": float(v.mean()), "cycles_sd": float(v.std(ddof=1)),
                                "ms_at_125MHz": float(v.mean() / 125e3), "per_image": v.tolist()}
            print(f"{k}: {v.mean() / 1e3:8.1f}k +/- {v.std(ddof=1) / 1e3:.1f}k cycles  "
                  f"= {v.mean() / 125e3:.2f} ms at 125 MHz")
    # ---- how well does the hand model (energy_model.CYC) predict the emulator?
    from energy_model import pico_cycles
    from intsim import quantise_snn_fast, snn_fast_int
    xs = x[:a.n]
    qa = quantise_ann(dict(np.load(os.path.join(WDIR, "ann_float.npz"))), xtr[:5000])
    _, h8, _ = ann_int(qa, xs)
    nz, nzh = (xs > 0).sum(1), (h8 > 0).sum(1)
    sl = input_spikes_int(xs, "latency", np.arange(a.n))
    _, _, Sl, _ = snn_int(quantise_snn(dict(np.load(os.path.join(WDIR, "snn_latency_float.npz")))), sl)
    pred = {"A": [pico_cycles("ann_dense", nz_in=n, nz_hid=m)["total"] for n, m in zip(nz, nzh)],
            "Z": [pico_cycles("ann_skip", nz_in=n, nz_hid=m)["total"] for n, m in zip(nz, nzh)],
            "L": [pico_cycles("snn_latency", events=e, hid_spikes=h)["total"]
                  for e, h in zip(sl.sum((1, 2)), Sl.sum((1, 2)))]}
    if has_fast:
        _, _, cnt = snn_fast_int(quantise_snn_fast(dict(np.load(os.path.join(WDIR, "snn_fast_float.npz")))), xs,
                                 exit_margin=fast_margin if fast_margin >= 0 else None)
        pred["E"] = [pico_cycles("snn_fast", events=e, hid_spikes=h, steps=st)["total"]
                     for e, h, st in zip(cnt["events"], cnt["hid_spikes"], cnt["steps"])]
    out["hand_model_error"] = {}
    for k, pv in pred.items():
        mv = np.array(rows[k], float)
        err = (np.array(pv) - mv) / mv
        out["hand_model_error"][k] = {"mean_pct": float(err.mean() * 100), "mean_abs_pct": float(np.abs(err).mean() * 100)}
        print(f"hand model vs emulator, {k}: mean error {err.mean() * 100:+.1f}%, mean |error| {np.abs(err).mean() * 100:.1f}%")

    # ---- the numbers the hologram shows as "predicted" until the INA226 has measured
    rep_path = os.path.join(WDIR, "export_report.json")
    accs = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    acc_key = {"A": "ann_int_acc", "Z": "ann_int_acc", "L": "snn_latency_int_acc", "P": "snn_poisson_int_acc",
               "E": "snn_fast_int_acc"}
    costs = {k: {"cycles": v["cycles_mean"], "accuracy": accs.get(acc_key[k])} for k, v in out["models"].items()}
    with open(os.path.join(ROOT, "pi", "model_costs.json"), "w") as f:
        json.dump(costs, f, indent=1)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "cycle_audit.json"), "w") as f:
        json.dump(out, f, indent=1)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
