# Conversation-app fork patches

Tracked snapshots of files we patch on `dwain-barnes/reachy_mini_conversation_app_local`
on the Pi (`/home/pollen/dn/conversation-app/...`) so they survive reinstalls
and are easy to upstream as a PR.

| File | What it fixes |
|---|---|
| `local_audio.py` | Adds `REMOTE_STT_ENDPOINT` and `REMOTE_TTS_ENDPOINT` env handling. Pi5 CPU is too slow for distil-whisper (>170 s for 2 s clip) and Kokoro (8x real-time); both ML models offload to the LAN host (jetson-assistant). |
| `openai_realtime.py` | Wires tool calling into the local-LLM path. The fork's local path was `# Call local LLM (no tool support - using base instruct model)` — model would emit `(play_emotion("happy"))` as plain text and motors never moved. Now passes `tools=[...]` from the existing registry, dispatches `tool_calls` via `dispatch_tool_call`, loops up to 3 hops. Also embeds tools in the system prompt as `<tools>...</tools>` (Qwen2.5-VL-NVFP4's chat template silently drops the OpenAI `tools=` field), trims long enum descriptions to fit the 4 K context, and drops temperature 0.7 → 0.4. |
| `prompts.py` | Appends an English-only constraint to every profile's instructions. Without it Qwen2.5-VL leaks Chinese/other scripts in short answers. |
| `tools/play_emotion.py` | Adds a friendly-name resolver. Qwen tends to ask for "happy"/"sad"/"curious" instead of the Pollen library names like "cheerful1"/"sad2"/"curious1"; the resolver maps via an alias table + suffix-numbered match before the strict equality check fires. |

## Applying

The Pi already has these. To re-apply (e.g. after a fresh `pip install`):

```bash
DST=/home/pollen/dn/conversation-app/src/reachy_mini_conversation_app
for f in local_audio.py openai_realtime.py prompts.py; do
  cp "$f" "$DST/$f"
done
cp play_emotion.py "$DST/tools/play_emotion.py"
```

## Required vLLM flags on the LLM host

Tool calling needs `--enable-auto-tool-choice --tool-call-parser hermes`.
See `docker-compose.thor-serve.yml` (vllm service `command:`).

## Required env vars (set in `configs/conversation-app.env`)

```
REMOTE_STT_ENDPOINT=http://<llm-host>:8080/stt/transcribe
REMOTE_TTS_ENDPOINT=http://<llm-host>:8080/tts/synthesize
```

When either env is unset, `LocalASR` / `LocalTTS` fall back to their original
on-device behavior — patches are non-destructive.
