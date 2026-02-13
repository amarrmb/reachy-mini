# ── Reachy Mini Voice Assistant — Jetson Thor ────────────────────────────
#
# Layers Reachy Mini robot tools on top of jetson-assistant.
# Includes: 9 robot tools, MotionManager, MuJoCo simulation.
#
# Build on Thor:
#   docker build -t reachy-mini:thor .
#
# Run with docker compose:
#   docker compose up -d       # starts vLLM + reachy assistant

ARG BASE_IMAGE=ghcr.io/amarrmb/jetson-assistant:thor
FROM ${BASE_IMAGE}

WORKDIR /app/reachy-mini
COPY . .

# Upgrade pip first (reachy-mini requires pip>=25, debian ships 24.0
# which can't be uninstalled cleanly, so use --ignore-installed)
RUN pip install --no-cache-dir --break-system-packages --ignore-installed pip>=25

# Reachy Mini SDK (includes MuJoCo for sim mode)
RUN pip install --no-cache-dir --break-system-packages \
    reachy-mini[mujoco] Pillow

# bot/ on PYTHONPATH so reachy_tools is importable
ENV PYTHONPATH="/app/reachy-mini/bot:${PYTHONPATH}"

ENTRYPOINT ["jetson-assistant"]
CMD ["assistant", "--config", "/app/reachy-mini/configs/default.yaml"]
