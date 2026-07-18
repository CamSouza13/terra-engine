#!/usr/bin/env bash
# Terra edge-node first-boot provisioning (Raspberry Pi OS Lite).
# Idempotent: safe to re-run. Brands the node, installs the engine, and enables
# the service. Run once as root: sudo bash firstboot.sh [domain]
set -euo pipefail

DOMAIN="${1:-aquaculture}"
REPO="https://github.com/CamSouza13/terra-engine"

echo ">> Terra node provisioning (domain=${DOMAIN})"

# 1. identity / branding
hostnamectl set-hostname "terra-node" || true
id -u terra >/dev/null 2>&1 || useradd -m -s /bin/bash terra
install -d -o terra -g terra /opt/terra /var/lib/terra

# 2. dependencies + engine (numpy-only edge install)
apt-get update -y
apt-get install -y python3 python3-pip python3-numpy git i2c-tools
if [ ! -d /opt/terra/terra-engine ]; then
  git clone "${REPO}" /opt/terra/terra-engine
fi
pip3 install -e /opt/terra/terra-engine --break-system-packages

# 3. enable I2C for smart sensors (Atlas EZO etc.)
raspi-config nonint do_i2c 0 || true

# 4. install + start the service
sed "s/--domain aquaculture/--domain ${DOMAIN}/g" \
  /opt/terra/terra-engine/deploy/terra-node.service \
  > /etc/systemd/system/terra-node.service
systemctl daemon-reload
systemctl enable --now terra-node.service

echo ">> Done. Status:"
systemctl --no-pager status terra-node.service || true
echo ">> Logs: journalctl -u terra-node -f"
