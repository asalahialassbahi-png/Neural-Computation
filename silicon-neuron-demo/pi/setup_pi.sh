#!/usr/bin/env bash
# One-time setup for the Raspberry Pi Zero 2 W that drives the hologram.
# Run on the Pi (Raspberry Pi OS Lite, Bookworm) from inside this pi/ folder:
#     chmod +x setup_pi.sh && ./setup_pi.sh && sudo reboot
set -euo pipefail

CFG=/boot/firmware/config.txt
CMD=/boot/firmware/cmdline.txt
[ -f "$CFG" ] || CFG=/boot/config.txt          # older images
[ -f "$CMD" ] || CMD=/boot/cmdline.txt

echo "== packages"
sudo apt-get update
sudo apt-get install -y python3-pygame python3-numpy python3-serial python3-smbus2 \
     python3-gpiozero python3-lgpio python3-scipy python3-matplotlib i2c-tools

echo "== UART for the Pico: full PL011 UART on GPIO14/15, no serial login console"
sudo raspi-config nonint do_serial_hw 0        # enable UART hardware
sudo raspi-config nonint do_serial_cons 1      # disable login shell on it
sudo raspi-config nonint do_i2c 0              # enable I2C for the INA226
add_line() { grep -qxF "$1" "$CFG" || echo "$1" | sudo tee -a "$CFG" >/dev/null; }
add_line "enable_uart=1"
add_line "dtoverlay=disable-bt"                # gives /dev/serial0 the good UART
add_line "dtparam=i2c_arm=on"
add_line "dtparam=i2c_arm_baudrate=400000"    # faster INA226 sampling

echo "== force 1024x600 on HDMI (7in panels sometimes send a poor EDID)"
grep -q "video=HDMI-A-1:1024x600@60" "$CMD" || sudo sed -i 's/$/ video=HDMI-A-1:1024x600@60/' "$CMD"

echo "== autostart the hologram at boot"
DIR="$(cd "$(dirname "$0")" && pwd)"
sed "s|__DIR__|$DIR|g; s|__USER__|$USER|g" "$DIR/hologram.service" | sudo tee /etc/systemd/system/hologram.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable hologram.service

echo "done — reboot, then check:  i2cdetect -y 1   (INA226 should show at 40)"
