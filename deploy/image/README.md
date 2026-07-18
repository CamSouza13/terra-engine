# Branded Terra node image

Assets and a recipe for turning a stock Raspberry Pi OS card into a Terra-branded
edge node: hostname, login banner, boot splash, and the engine running as a
service on first boot.

## Files

- `splash.svg` — the boot splash (closed-loop mark + wordmark). Rendered to PNG
  on the Pi by `setup-branding.sh`.
- `motd` — the SSH login banner.
- `setup-branding.sh` — applies hostname, MOTD, and splash.

## Bake it into a card

Two ways, from easiest to most repeatable.

**A. Raspberry Pi Imager + first boot.** Flash Raspberry Pi OS Lite, then on the
node run the provisioning script from the repo root:

```bash
sudo bash deploy/firstboot.sh aquaculture     # installs the engine + service
sudo bash deploy/image/setup-branding.sh       # applies the Terra branding
```

`firstboot.sh` clones/installs `terra-engine`, enables I2C, and starts the
`terra-node` service (see `deploy/README.md`). `setup-branding.sh` layers the
identity on top.

**B. pi-gen (repeatable image builds).** Add a custom stage that copies this repo
into the rootfs and runs `firstboot.sh` + `setup-branding.sh` in the stage's
`run.sh`, then build the `.img` with pi-gen. This produces a flashable,
pre-branded Terra node image for batch provisioning.

## Notes

- The splash needs `librsvg2-bin` (`rsvg-convert`) or `imagemagick` on the Pi to
  rasterise the SVG; `setup-branding.sh` skips the splash gracefully if neither
  is present.
- The engine itself is numpy-only and runs on a Pi Zero 2 W; calibration
  (jax/numpyro) stays off-node.
