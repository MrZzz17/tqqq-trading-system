#!/usr/bin/env bash
# Run on a fresh Oracle Cloud Ubuntu ARM instance (ssh as ubuntu).
# Usage:
#   curl -fsSL ... | bash   # or copy repo and: sudo bash deploy/oracle-cloud/bootstrap-ubuntu.sh
#
# Env overrides:
#   APP_ROOT=/home/ubuntu/tqqq-trading-system
#   GIT_REPO=https://github.com/MrZzz17/tqqq-trading-system.git
#   GIT_BRANCH=main

set -euo pipefail

APP_ROOT="${APP_ROOT:-/home/ubuntu/tqqq-trading-system}"
GIT_REPO="${GIT_REPO:-https://github.com/MrZzz17/tqqq-trading-system.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
DEPLOY_USER="${SUDO_USER:-ubuntu}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run with sudo: sudo bash $0"
  exit 1
fi

echo "==> Packages"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip git

echo "==> App checkout"
if [[ ! -d "${APP_ROOT}/.git" ]]; then
  sudo -u "${DEPLOY_USER}" git clone --branch "${GIT_BRANCH}" --depth 1 "${GIT_REPO}" "${APP_ROOT}"
else
  echo "Repo already present at ${APP_ROOT}; skipping clone."
fi

echo "==> Python venv"
sudo -u "${DEPLOY_USER}" python3 -m venv "${APP_ROOT}/.venv"
sudo -u "${DEPLOY_USER}" "${APP_ROOT}/.venv/bin/pip" install --upgrade pip wheel
sudo -u "${DEPLOY_USER}" "${APP_ROOT}/.venv/bin/pip" install -r "${APP_ROOT}/requirements-prod.txt"

echo "==> systemd"
SERVICE_SRC="${APP_ROOT}/deploy/oracle-cloud/streamlit-tqqq.service"
SERVICE_DST="/etc/systemd/system/streamlit-tqqq.service"
sed -e "s|DEPLOY_USER|${DEPLOY_USER}|g" \
    -e "s|DEPLOY_APP_ROOT|${APP_ROOT}|g" \
    "${SERVICE_SRC}" > "${SERVICE_DST}"
chmod 644 "${SERVICE_DST}"
systemctl daemon-reload
systemctl enable streamlit-tqqq.service
systemctl restart streamlit-tqqq.service

echo "==> UFW (if enabled)"
if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q 'Status: active'; then
  ufw allow 8501/tcp comment 'Streamlit TQQQ' || true
  ufw reload || true
fi

echo ""
echo "Bootstrap finished. Check:"
echo "  sudo systemctl status streamlit-tqqq"
echo "  curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8501/"
echo ""
echo "Open Oracle VCN security list + NSG: inbound TCP 8501 from your IP (or 0.0.0.0/0 for testing)."
echo "Then browse: http://YOUR_PUBLIC_IP:8501/"
