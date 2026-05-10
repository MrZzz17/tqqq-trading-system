# Deploy on Oracle Cloud (Always Free ARM)

This runs **your own** Streamlit process so the app does **not** use Streamlit Community Cloud sleep or wake-up delays. Stay within Oracle **Always Free eligible** shapes so the VM stays at **$0** (see [Oracle Always Free](https://docs.oracle.com/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)).

## 1. Create the VM (OCI Console)

1. Sign up for [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/).
2. **Compute → Instances → Create instance.**
3. **Image:** Ubuntu 22.04 (or 24.04) **aarch64** (ARM64).
4. **Shape:** **VM.Standard.A1.Flex** (Ampere). Pick an **Always Free eligible** configuration (for example **1 OCPU**, **6 GB RAM** keeps you inside typical free limits; adjust only using the capacity planner so it stays marked free).
5. **Networking:** Use the default VCN or create one. Ensure the instance gets a **public IP**.
6. **SSH key:** Add your public key so you can `ssh ubuntu@<public-ip>`.

## 2. Open the app port (required)

Streamlit listens on **8501** by default.

1. **VCN → Security lists** for the subnet → **Ingress rules** → Add:
   - **Source:** `0.0.0.0/0` (testing) or **your home IP**/32 (safer)
   - **IP protocol:** TCP
   - **Destination port:** `8501`
2. If you use a **Network Security Group** on the instance, add the **same rule** there.

Without this step, the browser will time out even if Streamlit is running.

## 3. Bootstrap the server

SSH in:

```bash
ssh ubuntu@YOUR_PUBLIC_IP
```

Install from your fork (recommended) or this repo:

```bash
sudo GIT_REPO=https://github.com/MrZzz17/tqqq-trading-system.git GIT_BRANCH=main \
  bash -c 'curl -fsSL https://raw.githubusercontent.com/MrZzz17/tqqq-trading-system/main/deploy/oracle-cloud/bootstrap-ubuntu.sh | bash'
```

Or clone first, then run the script from disk:

```bash
git clone https://github.com/MrZzz17/tqqq-trading-system.git
cd tqqq-trading-system
sudo bash deploy/oracle-cloud/bootstrap-ubuntu.sh
```

This installs dependencies into `.venv`, installs **`streamlit-tqqq.service`**, and starts it.

Check:

```bash
sudo systemctl status streamlit-tqqq
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8501/
```

Browse: **`http://YOUR_PUBLIC_IP:8501/`**

## 4. Updates after you push to GitHub

```bash
cd ~/tqqq-trading-system   # or APP_ROOT you used
git pull
sudo systemctl restart streamlit-tqqq
```

## 5. HTTPS (optional, later)

For TLS you can put **Caddy** or **nginx** on ports 80/443 as a reverse proxy to `127.0.0.1:8501`, open 80/443 in the security list, and use Let’s Encrypt. Not required for personal use.

## 6. TradingView env (optional)

If you set `TV_USERNAME` / `TV_PASSWORD` (or `TV_SESSION`) in `core/data.py`’s expectations, add them to the service:

```bash
sudo systemctl edit streamlit-tqqq
```

Use:

```ini
[Service]
Environment=TV_USERNAME=...
Environment=TV_PASSWORD=...
```

Then `sudo systemctl daemon-reload && sudo systemctl restart streamlit-tqqq`.

---

**Disclaimer:** Oracle billing and free-tier rules change; always confirm shapes and usage in your tenancy. This guide is operational only, not Oracle billing advice.
