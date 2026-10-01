"""
INA226 (preferred) / INA219 (fallback) high-side current monitor, read over I2C.

We read the raw shunt and bus voltages and apply Ohm's law ourselves
(I = V_shunt / R_shunt, P = V_bus * I) instead of trusting the chip's
calibration register, so every number in the dissertation traces back to one
formula and one measured resistor.

INA226 registers
    0x00 config   AVG[11:9] VBUSCT[8:6] VSHCT[5:3] MODE[2:0]
    0x01 shunt voltage, signed, LSB = 2.5 uV
    0x02 bus voltage, LSB = 1.25 mV
    0xFE manufacturer id = 0x5449 ("TI")
INA219 registers
    0x01 shunt voltage, signed, LSB = 10 uV
    0x02 bus voltage, bits 15..3, LSB = 4 mV
"""
import time

try:
    from smbus2 import SMBus, i2c_msg
except ImportError:            # allows import on a PC without smbus2
    SMBus = None

CT_CODES = {140: 0, 204: 1, 332: 2, 588: 3, 1100: 4, 2116: 5, 4156: 6, 8244: 7}
AVG_CODES = {1: 0, 4: 1, 16: 2, 64: 3, 128: 4, 256: 5, 512: 6, 1024: 7}


class PowerSensor:
    def __init__(self, chip="ina226", bus=1, addr=0x40, r_shunt=0.100,
                 conv_us=588, avg=1):
        if SMBus is None:
            raise RuntimeError("pip install smbus2")
        self.chip, self.addr, self.r = chip, addr, r_shunt
        self.bus = SMBus(bus)
        if chip == "ina226":
            mid = self._read_u16(0xFE)
            if mid != 0x5449:
                raise RuntimeError(f"INA226 not found at 0x{addr:02x} (id 0x{mid:04x})")
            ct = CT_CODES[conv_us]
            cfg = (AVG_CODES[avg] << 9) | (ct << 6) | (ct << 3) | 0b111
            self._write_u16(0x00, cfg)
            self.update_period_s = avg * 2 * conv_us * 1e-6
        else:
            # INA219: 32 V range, +/-40 mV PGA (/1), 12-bit, continuous shunt+bus
            self._write_u16(0x00, 0x2000 | (0 << 11) | (0x3 << 7) | (0x3 << 3) | 0b111)
            self.update_period_s = 2 * 532e-6
        time.sleep(0.01)

    def _read_u16(self, reg):
        w = i2c_msg.write(self.addr, [reg])
        r = i2c_msg.read(self.addr, 2)
        self.bus.i2c_rdwr(w, r)
        b = list(r)
        return (b[0] << 8) | b[1]

    def _write_u16(self, reg, val):
        self.bus.write_i2c_block_data(self.addr, reg, [(val >> 8) & 0xFF, val & 0xFF])

    def read(self):
        """Returns (V_bus [V], I [A], P [W])."""
        raw_s = self._read_u16(0x01)
        if raw_s & 0x8000:
            raw_s -= 1 << 16
        raw_b = self._read_u16(0x02)
        if self.chip == "ina226":
            v_shunt = raw_s * 2.5e-6
            v_bus = raw_b * 1.25e-3
        else:
            v_shunt = raw_s * 10e-6
            v_bus = (raw_b >> 3) * 4e-3
        i = v_shunt / self.r
        return v_bus, i, v_bus * i


class FakePowerSensor:
    """Stand-in used by --sim: idle power plus a load term while the marker is high."""

    def __init__(self, link):
        import random
        self.link, self.rand = link, random.Random(1)
        self.update_period_s = 0.001

    def read(self):
        base = 0.0931 + self.rand.gauss(0, 0.0004)       # ~18.6 mA at 5.0 V
        extra = self.link.fake_load_w() if self.link.marker() else 0.0
        v = 4.98 + self.rand.gauss(0, 0.002)
        p = base + extra
        return v, p / v, p
