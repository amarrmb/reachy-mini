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

FROM ghcr.io/amarrmb/jetson-assistant:thor

WORKDIR /app/reachy-mini
COPY . .

# Reachy Mini SDK (includes MuJoCo for sim mode)
RUN pip install --no-cache-dir --break-system-packages \
    reachy-mini[mujoco] Pillow

# bot/ on PYTHONPATH so reachy_tools is importable
ENV PYTHONPATH="/app/reachy-mini/bot:${PYTHONPATH}"

ENTRYPOINT ["jetson-assistant"]
CMD ["assistant", "--config", "/app/reachy-mini/configs/default.yaml"]
