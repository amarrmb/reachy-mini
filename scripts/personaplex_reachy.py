#!/usr/bin/env python3
"""PersonaPlex + Reachy Mini audio-reactive demo.

Runs the PersonaPlex full-duplex speech-to-speech server with Reachy Mini's
MotionManager wired to the model's audio output. The robot breathes, tilts
attentively while listening, and sways in sync with PersonaPlex's speech.

No tool calling. No jetson-assistant dependency. Just audio-reactive motions.

Usage:
    # With real Reachy Mini:
    python scripts/personaplex_reachy.py \\
        --personaplex-dir ~/baskd/personaplex-oss \\
        --port 8998 --fp8 \\
        --reachy-host 192.168.0.29

    # Standalone PersonaPlex (no robot):
    python scripts/personaplex_reachy.py \\
        --personaplex-dir ~/baskd/personaplex-oss \\
        --port 8998 --fp8
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="PersonaPlex + Reachy Mini audio-reactive demo"
    )

    # PersonaPlex args
    parser.add_argument(
        "--personaplex-dir", type=str, required=True,
        help="Path to personaplex-oss repo",
    )
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument("--ssl", type=str)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--tokenizer", type=str)

    # Reachy args
    parser.add_argument(
        "--reachy-host", type=str, default=None,
        help="Reachy Mini daemon IP. Omit to run PersonaPlex standalone.",
    )
    parser.add_argument(
        "--tick-rate", type=float, default=50.0,
        help="MotionManager update rate in Hz (default: 50)",
    )

    args = parser.parse_args()

    # ── Setup sys.path ──
    personaplex_dir = os.path.expanduser(args.personaplex_dir)
    if not os.path.isdir(personaplex_dir):
        print(f"Error: --personaplex-dir not found: {personaplex_dir}", file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, personaplex_dir)

    # Add reachy-mini/bot/ for motion_manager, personaplex_bridge, reachy_connect
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bot_dir = os.path.join(os.path.dirname(script_dir), "bot")
    sys.path.insert(0, bot_dir)

    # ── Imports (after path setup) ──
    import torch
    from aiohttp import web
    from huggingface_hub import hf_hub_download
    import sentencepiece

    from moshi.moshi.server import (
        ServerState,
        _get_static_path,
        _get_voice_prompt_dir,
        seed_all,
        torch_auto_device,
    )
    from moshi.moshi.models import loaders
    from moshi.moshi.utils.connection import create_ssl_context, get_lan_ip
    from moshi.moshi.utils.logging import setup_logger

    logger = setup_logger(__name__)

    # ── Connect Reachy + setup MotionManager ──
    mm = None
    on_audio_frame = None

    if args.reachy_host:
        from personaplex_bridge import create_audio_bridge, setup_motion_manager
        from reachy_connect import connect_reachy

        logger.info(f"Connecting to Reachy Mini at {args.reachy_host}...")
        reachy = connect_reachy(host=args.reachy_host)
        logger.info("Reachy Mini connected")

        mm = setup_motion_manager(lambda: reachy, tick_rate=args.tick_rate)
        on_audio_frame = create_audio_bridge(mm)
        logger.info(
            "MotionManager started (%.0fHz) with breathing + listening + audio_sway",
            args.tick_rate,
        )
    else:
        logger.info("No --reachy-host, running PersonaPlex standalone (no robot)")

    # ── Load PersonaPlex model ──
    device = torch_auto_device(args.device)
    seed_all(42424242)
    hf_repo = args.hf_repo or loaders.DEFAULT_REPO

    logger.info("Loading Mimi...")
    mimi_weight = args.mimi_weight or hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)

    # FP16 + compile for mimi (always — matches personaplex-oss defaults)
    mimi = mimi.half()
    other_mimi = other_mimi.half()
    mimi.torch_compile_encoder_decoder = True
    mimi = torch.compile(mimi)
    logger.info("Mimi loaded (FP16 + compiled)")

    tokenizer_path = args.tokenizer or hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    logger.info("Loading Moshi LM...")
    moshi_weight = args.moshi_weight or hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device)
    lm.eval()

    if args.fp8:
        from moshi.moshi.fp8_quantize import quantize_model
        logger.info("Applying FP8 quantization...")
        quantize_model(lm)
        logger.info("FP8 quantization complete")

    voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, hf_repo)

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=device,
        voice_prompt_dir=voice_prompt_dir,
        fp8=args.fp8,
        on_audio_frame=on_audio_frame,
    )

    logger.info("Warming up model...")
    state.warmup()

    if args.fp8:
        from moshi.moshi.fp8_quantize import free_bf16_inproj
        free_bf16_inproj(lm)

    # ── Start server ──
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)

    static_path = _get_static_path(args.static)
    if static_path:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    ssl_context = None
    protocol = "http"
    if args.ssl:
        ssl_context, protocol = create_ssl_context(args.ssl)

    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"PersonaPlex Web UI: {protocol}://{host_ip}:{args.port}")
    if args.reachy_host:
        logger.info(f"Reachy Mini: {args.reachy_host} (audio-reactive motions active)")

    try:
        with torch.no_grad():
            web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
    finally:
        if mm is not None:
            logger.info("Stopping MotionManager...")
            mm.stop()
        if args.reachy_host:
            try:
                reachy.goto_sleep()
            except Exception:
                pass
            try:
                reachy.disconnect()
            except Exception:
                pass
            logger.info("Reachy Mini disconnected")


if __name__ == "__main__":
    main()
