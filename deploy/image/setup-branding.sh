#!/usr/bin/env bash
# Apply Terra branding to a Raspberry Pi OS node: hostname, login MOTD, and the
# boot splash. Run as root on the Pi (firstboot.sh calls this, or run standalone:
#   sudo bash setup-branding.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. identity
hostnamectl set-hostname terra-node || true

# 2. login banner
install -m 644 "$HERE/motd" /etc/motd

# 3. boot splash (Plymouth). Rasterise the SVG if a converter is present.
SPLASH_PNG=/tmp/terra-splash.png
if command -v rsvg-convert >/dev/null 2>&1; then
  rsvg-convert -w 1280 -h 720 "$HERE/splash.svg" -o "$SPLASH_PNG"
elif command -v convert >/dev/null 2>&1; then
  convert -background black -resize 1280x720 "$HERE/splash.svg" "$SPLASH_PNG"
else
  echo "note: install librsvg2-bin or imagemagick to render the boot splash" >&2
fi
if [ -f "$SPLASH_PNG" ] && [ -d /usr/share/plymouth/themes/pix ]; then
  cp "$SPLASH_PNG" /usr/share/plymouth/themes/pix/splash.png
fi

echo "Terra branding applied: hostname=terra-node, MOTD, boot splash."
