import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime

import cv2
import aiohttp
import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError
from gradio_client import Client as GradioClient

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)
from reachy_mini_conversation_app.local_audio import LocalVAD, LocalASR, LocalTTS


logger = logging.getLogger(__name__)

OPEN_AI_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OPEN_AI_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000



def _sanitize_tool_result(result):
    """Defang giant binary/base64 fields before pushing into LLM context."""
    if isinstance(result, dict):
        out = {}
        for k, v in result.items():
            if isinstance(v, str) and (k in ("b64_im", "image", "frame") or len(v) > 500):
                out[k] = f"<{len(v)} chars elided>"
            else:
                out[k] = v
        return out
    if isinstance(result, str) and len(result) > 500:
        return result[:500] + "... [truncated]"
    return result
class OpenaiRealtimeHandler(AsyncStreamHandler):
    """An OpenAI realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=OPEN_AI_INPUT_SAMPLE_RATE,
        )

        # Override typing of the sample rates to match OpenAI's requirements
        self.output_sample_rate: Literal[24000] = self.output_sample_rate
        self.input_sample_rate: Literal[24000] = self.input_sample_rate

        self.deps = deps

        # Override type annotations for OpenAI strict typing (only for values used in API)
        self.output_sample_rate = OPEN_AI_OUTPUT_SAMPLE_RATE
        self.input_sample_rate = OPEN_AI_INPUT_SAMPLE_RATE

        self.connection: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.start_time = asyncio.get_event_loop().time()
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        # Track how the API key was provided (env vs textbox) and its value
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0  # sequence counter to prevent stale emissions
        self.partial_debounce_delay = 0.5  # seconds

        # Internal lifecycle flags
        self._shutdown_requested: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()

        # =====================================================================
        # LOCAL AUDIO COMPONENTS (for full local mode)
        # =====================================================================

        # Built-in Local VAD (always available)
        self._local_vad = LocalVAD(
            energy_threshold=config.VAD_ENERGY_THRESHOLD,
            silence_duration=config.VAD_SILENCE_DURATION,
            min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
            sample_rate=self.input_sample_rate,
        )
        self._audio_buffer: list[bytes] = []  # Buffer for audio during speech
        self._is_speech_active: bool = False
        self._vad_processing: bool = False  # Prevent concurrent processing

        # External VAD endpoint (optional - for smart turn detection)
        self._local_vad_endpoint: str | None = config.LOCAL_VAD_ENDPOINT
        if self._local_vad_endpoint:
            logger.info("External VAD enabled at %s", self._local_vad_endpoint)

        # Built-in Local ASR (Distil-Whisper - lightweight for edge)
        self._local_asr: LocalASR | None = None

        if config.FULL_LOCAL_MODE:
            self._local_asr = LocalASR(
                model_name=config.DISTIL_WHISPER_MODEL,
                language=config.WHISPER_LANGUAGE,
            )
            logger.info("Built-in ASR (Distil-Whisper) initialized: %s model", config.DISTIL_WHISPER_MODEL)

        # Built-in Local TTS (Kokoro via FastRTC - lightweight for edge)
        self._local_tts: LocalTTS | None = None

        if config.FULL_LOCAL_MODE:
            # Use built-in Kokoro via FastRTC
            self._local_tts = LocalTTS(
                output_sample_rate=self.output_sample_rate,
                voice=config.KOKORO_VOICE,
                speed=config.KOKORO_SPEED,
            )
            logger.info("Built-in TTS initialized: Kokoro via FastRTC (voice: %s)", config.KOKORO_VOICE)

        # =====================================================================
        # LOCAL LLM CLIENT (LM Studio, Ollama, or vLLM)
        # =====================================================================
        self._local_llm_client: AsyncOpenAI | None = None
        self._local_llm_model: str = config.LOCAL_LLM_MODEL or "local-model"
        self._local_llm_provider: str = config.LLM_PROVIDER or "vllm"
        self._conversation_history: list[dict[str, Any]] = []
        self._pending_response_id: str | None = None  # Track OpenAI response to cancel

        if config.LOCAL_LLM_ENDPOINT:
            try:
                # Both LM Studio and Ollama use OpenAI-compatible APIs
                self._local_llm_client = AsyncOpenAI(
                    base_url=config.LOCAL_LLM_ENDPOINT,
                    api_key="not-needed",  # Local LLMs don't require API key
                )
                provider_name = config.LLM_PROVIDER.upper() if config.LLM_PROVIDER else "Local LLM"
                logger.info("%s client initialized at %s with model %s",
                           provider_name, config.LOCAL_LLM_ENDPOINT, self._local_llm_model)
            except Exception as e:
                logger.error("Failed to initialize local LLM client: %s", e)
                self._local_llm_client = None

        # Log if full local mode is enabled
        if self._is_full_local_mode:
            logger.info("=" * 60)
            logger.info("FULL LOCAL MODE: No OpenAI connection required")
            logger.info("=" * 60)

    @property
    def _is_full_local_mode(self) -> bool:
        """Check if we're in full local mode (no data sent to OpenAI)."""
        # Always True - fully local operation only
        return True

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    def _split_into_chunks(self, text: str, max_chars: int = 150) -> list[str]:
        """Split text into optimal chunks for TTS streaming.

        Uses a waterfall approach inspired by mlx-audio:
        1. First try to split at sentence boundaries (.!?…)
        2. Then try clause boundaries (:;)
        3. Then try phrase boundaries (,—)
        4. Finally fall back to space boundaries

        Args:
            text: The text to split.
            max_chars: Maximum characters per chunk (default 250 for fast response).

        Returns:
            List of text chunks to synthesize separately.
        """
        import re

        text = text.strip()
        if not text:
            return []

        # If text is short enough, return as-is
        if len(text) <= max_chars:
            return [text]

        chunks = []
        remaining = text

        # Waterfall punctuation priorities (strongest to weakest break points)
        waterfall = [
            r'([.!?…]+[\"\'\)]?\s+)',  # Sentence endings (with optional quotes/parens)
            r'([:;]\s+)',               # Clause separators
            r'([,—]\s+)',               # Phrase separators
            r'(\s+)',                   # Any whitespace (last resort)
        ]

        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining.strip())
                break

            # Try each punctuation level to find a good break point
            best_break = None
            for pattern in waterfall:
                # Find all matches within the max_chars window
                matches = list(re.finditer(pattern, remaining[:max_chars + 50]))
                if matches:
                    # Take the last match that's within or close to max_chars
                    for match in reversed(matches):
                        if match.end() <= max_chars + 20:  # Allow slight overflow for natural breaks
                            best_break = match.end()
                            break
                    if best_break:
                        break

            if best_break and best_break > 20:  # Don't create tiny chunks
                chunk = remaining[:best_break].strip()
                remaining = remaining[best_break:].strip()
            else:
                # No good break point found, force break at max_chars
                # Try to at least break at a space
                space_idx = remaining[:max_chars].rfind(' ')
                if space_idx > 20:
                    chunk = remaining[:space_idx].strip()
                    remaining = remaining[space_idx:].strip()
                else:
                    chunk = remaining[:max_chars].strip()
                    remaining = remaining[max_chars:].strip()

            if chunk:
                chunks.append(chunk)

        return chunks

    async def _synthesize_with_chatterbox(self, text: str) -> None:
        """Synthesize text using Chatterbox TTS and stream audio for immediate playback.

        Uses waterfall chunking to split text at natural boundaries for faster
        time-to-first-audio while maintaining natural speech flow.

        Args:
            text: The text to synthesize.

        """
        if not self._chatterbox_client:
            logger.warning("Chatterbox client not configured, skipping TTS")
            return

        # Split into optimal chunks using waterfall approach
        chunks = self._split_into_chunks(text)
        logger.info("TTS: input text (%d chars): %s", len(text), text[:100])
        logger.info("TTS: split into %d chunks: %s", len(chunks), [c[:30] + "..." if len(c) > 30 else c for c in chunks])

        for i, chunk in enumerate(chunks):
            logger.info("TTS: synthesizing chunk %d/%d: %s", i + 1, len(chunks), chunk[:50])
            await self._synthesize_sentence(chunk)

    async def _synthesize_sentence(self, text: str) -> None:
        """Synthesize a single sentence using Chatterbox Gradio endpoint.

        Args:
            text: The sentence to synthesize.
        """
        if not self._chatterbox_client:
            return

        try:
            logger.debug("TTS for sentence: %s", text[:50])

            # Run the blocking Gradio client call in a thread pool
            from gradio_client import handle_file

            # Wrap the reference audio file for Gradio
            ref_audio = handle_file(self._chatterbox_ref_audio) if self._chatterbox_ref_audio else None

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._chatterbox_client.predict(
                    text,                    # text
                    ref_audio,               # audio_prompt_path (wrapped)
                    0.8,                     # temperature
                    0,                       # seed_num
                    0.0,                     # min_p
                    0.95,                    # top_p
                    1000,                    # top_k
                    1.2,                     # repetition_penalty
                    True,                    # norm_loudness
                    fn_index=9,
                ),
            )

            # Handle different Gradio return formats
            if isinstance(result, str):
                # It's a file path - read the audio file
                import scipy.io.wavfile as wavfile
                sample_rate, audio_data = wavfile.read(result)
            elif isinstance(result, tuple) and len(result) >= 2:
                sample_rate, audio_data = result[0], result[1]
            elif isinstance(result, np.ndarray):
                sample_rate = 24000
                audio_data = result
            else:
                logger.error("Unexpected Chatterbox result format: %s", type(result))
                return

            # Convert to numpy array if needed
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)

            # Resample to output sample rate if different
            if sample_rate != self.output_sample_rate:
                num_samples = int(len(audio_data) * self.output_sample_rate / sample_rate)
                audio_data = resample(audio_data, num_samples)

            # Convert to int16
            audio_data = audio_to_int16(audio_data)

            # Feed to head wobbler if available
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(base64.b64encode(audio_data.tobytes()).decode("utf-8"))

            # Queue audio in chunks for smoother playback
            chunk_size = 4800  # 200ms at 24kHz
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await self.output_queue.put(
                    (self.output_sample_rate, chunk.reshape(1, -1)),
                )

            logger.debug("TTS sentence complete: %s", text[:30])

        except Exception as e:
            logger.error("Chatterbox TTS synthesis failed: %s", e)

    async def _check_turn_complete(self, audio_data: bytes) -> bool:
        """Check if the user's turn is complete using local VAD.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, 24kHz, mono)

        Returns:
            True if turn is complete, False if user might still be speaking
        """
        if not self._local_vad_endpoint:
            return True  # No VAD configured, assume complete

        try:
            import aiohttp
            import tempfile
            import wave

            # Save audio to temp WAV file for the VAD server
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(f, 'wb') as wav:
                    wav.setnchannels(1)  # mono
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(self.input_sample_rate)  # 24kHz
                    wav.writeframes(audio_data)

            # Read the WAV file and encode as base64
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()

            import base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            # Clean up temp file
            try:
                import os
                os.unlink(temp_path)
            except Exception:
                pass

            # Call VAD endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._local_vad_endpoint}/predict",
                    json={"audio_base64": audio_b64},
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        prediction = result.get("prediction", 1)
                        probability = result.get("probability", 1.0)
                        status = result.get("status", "complete")
                        logger.info("VAD result: %s (probability=%.2f)", status, probability)
                        return prediction == 1  # 1 = complete, 0 = incomplete
                    else:
                        logger.warning("VAD request failed with status %d", resp.status)
                        return True  # Assume complete on error

        except Exception as e:
            logger.error("VAD check failed: %s", e)
            return True  # Assume complete on error

    async def _transcribe_with_local_asr(self, audio_data: bytes) -> str | None:
        """Transcribe audio using local ASR (Distil-Whisper).

        Args:
            audio_data: Raw PCM audio bytes (16-bit, 24kHz, mono)

        Returns:
            Transcribed text or None if failed
        """
        # Use built-in Distil-Whisper
        if self._local_asr:
            try:
                transcript = await self._local_asr.transcribe(audio_data, self.input_sample_rate)
                if transcript:
                    return transcript
                logger.warning("Built-in ASR returned empty result")
                return None
            except Exception as e:
                logger.error("Built-in ASR transcription failed: %s", e)
                return None

        # Fall back to external Gradio ASR
        if not self._local_asr_client:
            logger.warning("No ASR provider available")
            return None

        try:
            import tempfile
            import wave

            # Save audio buffer to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(f, 'wb') as wav:
                    wav.setnchannels(1)  # mono
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(self.input_sample_rate)  # 24kHz
                    wav.writeframes(audio_data)

            logger.debug("Saved audio buffer to %s (%d bytes)", temp_path, len(audio_data))

            # Call external ASR via Gradio
            from gradio_client import handle_file
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._local_asr_client.predict(
                    handle_file(temp_path),  # Wrap file path for Gradio
                    fn_index=0,  # First (and only) function in the Interface
                )
            )

            # Clean up temp file
            try:
                import os
                os.unlink(temp_path)
            except Exception:
                pass

            if isinstance(result, str) and result.strip():
                transcript = result.strip()
                # Filter out error messages
                if transcript.startswith("[") and transcript.endswith("]"):
                    logger.warning("ASR returned placeholder: %s", transcript)
                    return None
                logger.info("External ASR transcription: %s", transcript)
                return transcript

            logger.warning("External ASR returned empty result")
            return None

        except Exception as e:
            logger.error("External ASR transcription failed: %s", e)
            return None

    async def _process_local_asr(self, audio_data: bytes, check_vad: bool = True) -> None:
        """Process audio with local ASR and generate response with local LLM.

        Args:
            audio_data: Raw PCM audio bytes from the speech buffer.
            check_vad: Whether to check VAD first (default True).
        """
        # Check if turn is complete using local VAD
        if check_vad and self._local_vad_endpoint:
            is_complete = await self._check_turn_complete(audio_data)
            if not is_complete:
                logger.info("VAD says turn incomplete - waiting for more speech")
                # Re-enable speech buffering to capture more audio
                self._is_speech_active = True
                # Put the audio back in the buffer
                self._audio_buffer.append(audio_data)
                return

        # Transcribe with local ASR
        transcript = await self._transcribe_with_local_asr(audio_data)
        if not transcript:
            logger.warning("Local ASR returned no transcription")
            return

        # Show transcription in UI
        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

        # Generate response with local LLM
        if self._local_llm_client:
            await self._generate_local_response(transcript)
        else:
            logger.warning("Local LLM not available, cannot generate response")

    async def _generate_local_response(self, user_message: str) -> None:
        """Generate a response using the local LLM and send to Chatterbox.

        Args:
            user_message: The user's transcribed message.

        """
        if not self._local_llm_client:
            logger.warning("Local LLM client not available")
            return

        try:
            # Add user message to conversation history
            self._conversation_history.append({"role": "user", "content": user_message})

            # Build the system prompt. Qwen2.5-VL-NVFP4's chat template silently
            # drops the OpenAI tools= field, so the model never sees its tools. We
            # work around this by embedding the tool list in the system message in
            # Qwen's native <tools>/<tool_call> format. The hermes parser
            # configured on vLLM decodes the <tool_call>...</tool_call> blocks back
            # into structured tool_calls.
            base_instructions = get_session_instructions()
            try:
                from reachy_mini_conversation_app.tools.core_tools import ALL_TOOL_SPECS as _SPECS
            except Exception:
                _SPECS = []
            if _SPECS:
                import json as _json
                # Strip long enum/catalog text from parameter descriptions so
                # we don't blow Qwen's 4 K context. play_emotion alone has
                # ~9 KB of inline emotion descriptions; the dispatcher
                # accepts any name so the model doesn't need the full
                # catalog in-prompt.
                _NL = chr(10)
                def _slim(spec):
                    params = spec.get("parameters") or {"type": "object", "properties": {}}
                    props = {}
                    for pname, p in (params.get("properties") or {}).items():
                        slim = {k: v for k, v in p.items() if k in ("type", "enum", "items")}
                        desc = " ".join((p.get("description") or "").split())
                        if len(desc) > 1200:
                            desc = desc[:1200].rstrip() + "..."
                        if desc:
                            slim["description"] = desc
                        if isinstance(slim.get("enum"), list) and len(slim["enum"]) > 12:
                            slim["enum"] = slim["enum"][:12] + ["..."]
                        props[pname] = slim
                    return {
                        "type": "function",
                        "function": {
                            "name": spec["name"],
                            "description": (spec.get("description") or "").strip(),
                            "parameters": {
                                "type": "object",
                                "properties": props,
                                "required": params.get("required", []),
                            },
                        },
                    }
                tools_block_lines = [_json.dumps(_slim(s)) for s in _SPECS]
                tools_block = "\n".join(tools_block_lines)
                tool_preamble = (
                    "You are an embodied tabletop robot named Reachy. You have a "
                    "physical body with a head, antennas, and a camera. You are "
                    "physically present with the user.\n\n"
                    "When the user asks you to do ANY physical action — move, "
                    "dance, look around, wave, play an emotion, take a photo, "
                    "stop dancing — you MUST call the matching tool. Do not just "
                    "say you will do it. Do not describe the action in words. "
                    "CALL THE TOOL.\n\n"
                    "Examples:\n"
                    "- User: \"dance\"  -> CALL dance() then say \"Dancing!\"\n"
                    "- User: \"can you do a dance\" -> CALL dance() then say \"Sure!\"\n"
                    "- User: \"look up\" -> CALL move_head(direction=\"up\") then say \"Looking up.\"\n"
                    "- User: \"play happy\" -> CALL play_emotion(emotion=\"happy\") then say \"Feeling happy!\"\n"
                    "- User: \"wave at me\" -> CALL play_emotion(emotion=\"wave\") then say \"Hi!\"\n"
                    "- User: \"tell me a joke\" -> just answer in text, no tool needed.\n"
                    "- User: \"thank you\" / \"hi\" / \"hello\" -> just greet in text, NO TOOL.\n\n"
                    "Never put emoji or unicode pictographs in your spoken response — "
                    "they are read aloud as words.\n\n"
                    "# Tools\n\n"
                    "You are provided with function signatures within "
                    "<tools></tools> XML tags:\n<tools>\n" + tools_block + "\n</tools>\n\n"
                    "For each function call, return a json object with the function "
                    "name and arguments within <tool_call></tool_call> XML tags:\n"
                    "<tool_call>\n{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}\n</tool_call>"
                )
                system_content = base_instructions + "\n\n" + tool_preamble
            else:
                system_content = base_instructions
            messages = [
                {"role": "system", "content": system_content},
                *self._conversation_history[-20:]  # Keep last 20 messages for context
            ]

            logger.debug("Calling local LLM with %d messages", len(messages))

            # Build OpenAI-shape tool list from the existing registry. The fork
            # only wired tools into the OpenAI Realtime path; in FULL_LOCAL_MODE
            # we now pass them to vLLM and dispatch any tool_calls the model
            # emits via dispatch_tool_call (same dispatcher the OpenAI path uses).
            try:
                from reachy_mini_conversation_app.tools.core_tools import ALL_TOOL_SPECS, dispatch_tool_call
                openai_tools = [
                    {"type": "function", "function": {
                        "name": s["name"],
                        "description": s.get("description", ""),
                        "parameters": s.get("parameters", {"type": "object", "properties": {}}),
                    }} for s in ALL_TOOL_SPECS
                ]
            except Exception as _e:
                logger.warning("Tool registry unavailable: %s", _e)
                openai_tools = []
                dispatch_tool_call = None

            # Tool-call loop: at most 3 hops (model can chain tools).
            text_response = None
            for _hop in range(3):
                kwargs = dict(model=self._local_llm_model, messages=messages,
                              max_tokens=512, temperature=0.2)
                if openai_tools:
                    kwargs["tools"] = openai_tools
                    kwargs["tool_choice"] = "auto"
                response = await self._local_llm_client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                assistant_message = choice.message
                tool_calls = getattr(assistant_message, "tool_calls", None) or []
                if tool_calls and dispatch_tool_call:
                    # Append the assistant turn (with tool_calls) into history.
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name,
                                          "arguments": tc.function.arguments}}
                            for tc in tool_calls
                        ],
                    })
                    for tc in tool_calls:
                        try:
                            tool_result = await dispatch_tool_call(
                                tc.function.name, tc.function.arguments or "{}", self.deps
                            )
                            logger.info("Tool '%s' executed -> %s", tc.function.name, str(tool_result)[:120])
                        except Exception as e:
                            logger.error("Tool '%s' dispatch failed: %s", tc.function.name, e)
                            tool_result = {"error": str(e)}
                        import json as _json
                        _safe_result = _sanitize_tool_result(tool_result)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": _json.dumps(_safe_result, default=str),
                        })
                        await self.output_queue.put(AdditionalOutputs({
                            "role": "assistant",
                            "content": _json.dumps(_safe_result, default=str),
                            "metadata": {"title": f"🛠️ Used tool {tc.function.name}", "status": "done"},
                        }))
                    continue  # Loop again so the model can speak after the tool result.
                # No tool calls: take the text response and exit the loop.
                text_response = assistant_message.content
                break
            else:
                logger.warning("Tool-call loop hit 3-hop cap without final text response")

            if text_response:
                # Clean up thinking tags from various models (Qwen, DeepSeek, etc.)
                import re
                # Remove <think>...</think> tags (Qwen style)
                if "<think>" in text_response:
                    text_response = re.sub(r'<think>.*?</think>', '', text_response, flags=re.DOTALL).strip()
                # Remove <thinking>...</thinking> tags (other models)
                if "<thinking>" in text_response:
                    text_response = re.sub(r'<thinking>.*?</thinking>', '', text_response, flags=re.DOTALL).strip()

                logger.info("Local LLM response: %s", text_response[:100])

                # Add to conversation history
                self._conversation_history.append({"role": "assistant", "content": text_response})

                # Show in UI
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": text_response})
                )

                # Synthesize with local TTS
                await self._synthesize_locally(text_response)

        except Exception as e:
            logger.error("Local LLM generation failed: %s", e)
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": f"[error] LLM failed: {e}"})
            )

    async def _synthesize_locally(self, text: str) -> None:
        """Synthesize text using the configured local TTS provider.

        Args:
            text: The text to synthesize.
        """
        if not text or not text.strip():
            return

        # Use built-in local TTS (Kokoro via FastRTC)
        if self._local_tts:
            try:
                audio_data = await self._local_tts.synthesize(text)
                if audio_data is not None:
                    # Feed to head wobbler if available
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.feed(base64.b64encode(audio_data.tobytes()).decode("utf-8"))

                    # Queue audio in chunks for smoother playback
                    chunk_size = 4800  # 200ms at 24kHz
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i : i + chunk_size]
                        await self.output_queue.put(
                            (self.output_sample_rate, chunk.reshape(1, -1)),
                        )
                    logger.debug("Local TTS synthesis complete")
                else:
                    logger.warning("Local TTS returned no audio")
            except Exception as e:
                logger.error("Local TTS synthesis failed: %s", e)
        else:
            logger.warning("No TTS provider available")

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        - Updates the global config's selected profile for subsequent calls.
        - If a realtime connection is active, sends a session.update with the
          freshly resolved instructions so the change takes effect immediately.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = get_session_instructions()
                voice = get_session_voice()
            except BaseException as e:  # catch SystemExit from prompt loader without crashing
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Attempt a live update first, then force a full restart to ensure it sticks
            if self.connection is not None:
                try:
                    await self.connection.session.update(
                        session={
                            "type": "realtime",
                            "instructions": instructions,
                            "audio": {"output": {"voice": voice}},
                        },
                    )
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

                # Force a real restart to guarantee the new instructions/voice
                try:
                    await self._restart_session()
                    return "Applied personality and restarted realtime session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                logger.info(
                    "Applied personality recorded: %s (no live connection; will apply on next session)",
                    profile or "built-in default",
                )
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _emit_debounced_partial(self, transcript: str, sequence: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)
            # Only emit if this is still the latest partial (by sequence number)
            if self.partial_transcript_sequence == sequence:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        # In full local mode, skip OpenAI entirely
        if self._is_full_local_mode:
            logger.info("Starting in FULL LOCAL MODE - no OpenAI connection needed")
            await self._run_local_only_session()
            return

        openai_api_key = config.OPENAI_API_KEY
        if self.gradio_mode and not openai_api_key:
            # api key was not found in .env or in the environment variables
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_api_key = args[3] if len(args[3]) > 0 else None
            if textbox_api_key is not None:
                openai_api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
            else:
                openai_api_key = config.OPENAI_API_KEY
        else:
            if not openai_api_key or not openai_api_key.strip():
                # In headless console mode, LocalStream now blocks startup until the key is provided.
                # However, unit tests may invoke this handler directly with a stubbed client.
                # To keep tests hermetic without requiring a real key, fall back to a placeholder.
                logger.warning("OPENAI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
                openai_api_key = "DUMMY"

        self.client = AsyncOpenAI(api_key=openai_api_key)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                # Normal exit from the session, stop retrying
                return
            except ConnectionClosedError as e:
                # Abrupt close (e.g., "no close frame received or sent") → retry
                logger.warning("Realtime websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, etc.
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                # never keep a stale reference
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: OpenAI client not initialized yet.")
                return

            # Fire-and-forget new session and wait briefly for connection
            try:
                self._connected_event.clear()
            except Exception:
                pass
            asyncio.create_task(self._run_realtime_session(), name="openai-realtime-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _run_local_only_session(self) -> None:
        """Run in full local mode without any OpenAI connection.

        This handles the entire pipeline locally:
        - Energy-based VAD for speech start detection
        - Smart-turn VAD for turn completion
        - Local ASR (GLM-ASR-Nano)
        - Local LLM (Qwen via vLLM)
        - Local TTS (Chatterbox)
        """
        logger.info("Local-only session started - VAD, ASR, LLM, and TTS all running locally")

        # Signal that we're ready to receive audio
        self._connected_event.set()

        # The audio processing happens in receive() which is called by the audio input stream
        # We just need to keep this session alive
        while not self._shutdown_requested:
            await asyncio.sleep(0.1)

        logger.info("Local-only session ended")

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        async with self.client.realtime.connect(model=config.MODEL_NAME) as conn:
            try:
                # Build session config - conditionally include audio output based on Chatterbox
                audio_config: dict[str, Any] = {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.input_sample_rate,
                        },
                        "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                        "turn_detection": {
                            "type": "server_vad",
                            "interrupt_response": True,
                        },
                    },
                }

                # Only include audio output if NOT using Chatterbox TTS
                if not self._chatterbox_client:
                    audio_config["output"] = {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.output_sample_rate,
                        },
                        "voice": get_session_voice(),
                    }

                session_config: dict[str, Any] = {
                    "type": "realtime",
                    "instructions": get_session_instructions(),
                    "audio": audio_config,
                    "tools": get_tool_specs(),
                    "tool_choice": "auto",
                }

                # When using local LLM, configure OpenAI for transcription only (no response generation)
                if self._local_llm_client:
                    # Minimal instructions since we won't use OpenAI's responses
                    session_config["instructions"] = "You are a transcription service. Do not respond."
                    # No tools - local LLM handles tool calls
                    session_config["tools"] = []
                    session_config["tool_choice"] = "none"
                    # Remove audio output config - we don't want OpenAI to speak
                    if "output" in session_config.get("audio", {}):
                        del session_config["audio"]["output"]
                    logger.info("Local LLM enabled - OpenAI transcription-only mode, local LLM will generate responses")
                # When using Chatterbox only (no local LLM), keep OpenAI for LLM but use Chatterbox for TTS
                elif self._chatterbox_client:
                    # Remove audio output - Chatterbox will handle TTS
                    if "output" in session_config.get("audio", {}):
                        del session_config["audio"]["output"]
                    logger.info("Chatterbox TTS enabled - OpenAI LLM mode, Chatterbox will synthesize audio")

                await conn.session.update(session=session_config)
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    get_session_voice(),
                )
                # If we reached here, the session update succeeded which implies the API key worked.
                # Persist the key to a newly created .env (copied from .env.example) if needed.
                self._persist_api_key_if_needed()
            except Exception:
                logger.exception("Realtime session.update failed; aborting startup")
                return

            logger.info("Realtime session updated successfully")

            # Manage event received from the openai server
            self.connection = conn
            try:
                self._connected_event.set()
            except Exception:
                pass
            async for event in self.connection:
                logger.debug(f"OpenAI event: {event.type}")
                if event.type == "input_audio_buffer.speech_started":
                    if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                        self._clear_queue()
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()
                    self.deps.movement_manager.set_listening(True)
                    # Start buffering for local ASR
                    if self._local_asr_client:
                        self._is_speech_active = True
                        self._audio_buffer.clear()
                        logger.debug("User speech started - buffering for local ASR")
                    else:
                        logger.debug("User speech started")

                if event.type == "input_audio_buffer.speech_stopped":
                    self.deps.movement_manager.set_listening(False)
                    # If using local ASR, transcribe the buffered audio
                    if self._local_asr_client and self._is_speech_active:
                        self._is_speech_active = False
                        if self._audio_buffer:
                            audio_data = b''.join(self._audio_buffer)
                            self._audio_buffer.clear()
                            logger.info("User speech stopped - transcribing %d bytes with local ASR", len(audio_data))
                            # Transcribe and generate response
                            asyncio.create_task(self._process_local_asr(audio_data))
                        else:
                            logger.debug("User speech stopped - no audio buffered")
                    else:
                        logger.debug("User speech stopped - server will auto-commit with VAD")

                if event.type in (
                    "response.audio.done",  # GA
                    "response.output_audio.done",  # GA alias
                    "response.audio.completed",  # legacy (for safety)
                    "response.completed",  # text-only completion
                ):
                    logger.debug("response completed")

                if event.type == "response.created":
                    logger.debug("Response created")
                    # When using local LLM, cancel OpenAI's auto-response
                    if self._local_llm_client:
                        response_id = getattr(event, "response", {})
                        if hasattr(response_id, "id"):
                            response_id = response_id.id
                        elif isinstance(response_id, dict):
                            response_id = response_id.get("id")
                        if response_id:
                            logger.debug("Cancelling OpenAI auto-response: %s", response_id)
                            try:
                                await self.connection.response.cancel()
                            except Exception as e:
                                logger.debug("Failed to cancel response (may already be done): %s", e)

                if event.type == "response.done":
                    # Doesn't mean the audio is done playing
                    logger.debug("Response done")

                # Handle partial transcription (user speaking in real-time) - skip if using local ASR
                if event.type == "conversation.item.input_audio_transcription.partial":
                    if self._local_asr_client:
                        continue  # Skip OpenAI transcription when using local ASR
                    logger.debug(f"User partial transcript: {event.transcript}")

                    # Increment sequence
                    self.partial_transcript_sequence += 1
                    current_sequence = self.partial_transcript_sequence

                    # Cancel previous debounce task if it exists
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass

                    # Start new debounce timer with sequence number
                    self.partial_transcript_task = asyncio.create_task(
                        self._emit_debounced_partial(event.transcript, current_sequence)
                    )

                # Handle completed transcription (user finished speaking) - skip if using local ASR
                if event.type == "conversation.item.input_audio_transcription.completed":
                    if self._local_asr_client:
                        logger.debug("Skipping OpenAI transcription (using local ASR)")
                        continue  # Skip OpenAI transcription when using local ASR
                    logger.debug(f"User transcript: {event.transcript}")

                    # Cancel any pending partial emission
                    if self.partial_transcript_task and not self.partial_transcript_task.done():
                        self.partial_transcript_task.cancel()
                        try:
                            await self.partial_transcript_task
                        except asyncio.CancelledError:
                            pass

                    await self.output_queue.put(AdditionalOutputs({"role": "user", "content": event.transcript}))

                    # If using local LLM, generate response locally instead of waiting for OpenAI
                    if self._local_llm_client and event.transcript and event.transcript.strip():
                        logger.info("Using local LLM for response generation")
                        asyncio.create_task(self._generate_local_response(event.transcript))

                # Handle assistant transcription (skip if using local LLM - we handle responses ourselves)
                if event.type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                    if self._local_llm_client:
                        logger.debug("Skipping OpenAI assistant transcript (using local LLM)")
                        continue
                    logger.debug(f"Assistant transcript: {event.transcript}")
                    await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": event.transcript}))

                    # If using Chatterbox TTS, synthesize the transcript
                    if self._chatterbox_client and event.transcript:
                        asyncio.create_task(self._synthesize_with_chatterbox(event.transcript))

                # Handle text-only responses (skip if using local LLM)
                if event.type in (
                    "response.text.done",
                    "response.output_text.done",
                    "response.content_part.done",
                ):
                    if self._local_llm_client:
                        logger.debug("Skipping OpenAI text response (using local LLM)")
                        continue
                    # Only process if using Chatterbox without local LLM
                    if self._chatterbox_client:
                        # Extract text from various event formats
                        text = getattr(event, "text", None) or getattr(event, "content", None)
                        if isinstance(text, dict):
                            text = text.get("text")
                        if text and isinstance(text, str):
                            logger.debug(f"Assistant text response: {text}")
                            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))
                            asyncio.create_task(self._synthesize_with_chatterbox(text))

                # Handle audio delta (skip if using Chatterbox TTS)
                if event.type in ("response.audio.delta", "response.output_audio.delta"):
                    if self._chatterbox_client:
                        # Skip OpenAI audio when using Chatterbox TTS
                        continue

                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.feed(event.delta)
                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(1, -1),
                        ),
                    )

                # ---- tool-calling plumbing (skip if using local LLM - it handles tools itself) ----
                if event.type == "response.function_call_arguments.done":
                    if self._local_llm_client:
                        logger.debug("Skipping OpenAI tool call (using local LLM)")
                        continue
                    tool_name = getattr(event, "name", None)
                    args_json_str = getattr(event, "arguments", None)
                    call_id = getattr(event, "call_id", None)

                    if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                        logger.error("Invalid tool call: tool_name=%s, args=%s", tool_name, args_json_str)
                        continue

                    try:
                        tool_result = await dispatch_tool_call(tool_name, args_json_str, self.deps)
                        logger.debug("Tool '%s' executed successfully", tool_name)
                        logger.debug("Tool result: %s", tool_result)
                    except Exception as e:
                        logger.error("Tool '%s' failed", tool_name)
                        tool_result = {"error": str(e)}

                    # send the tool result back
                    if isinstance(call_id, str):
                        await self.connection.conversation.item.create(
                            item={
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps(tool_result),
                            },
                        )

                    await self.output_queue.put(
                        AdditionalOutputs(
                            {
                                "role": "assistant",
                                "content": json.dumps(tool_result),
                                "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                            },
                        ),
                    )

                    if tool_name == "camera" and "b64_im" in tool_result:
                        # use raw base64, don't json.dumps (which adds quotes)
                        b64_im = tool_result["b64_im"]
                        if not isinstance(b64_im, str):
                            logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                            b64_im = str(b64_im)
                        await self.connection.conversation.item.create(
                            item={
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{b64_im}",
                                    },
                                ],
                            },
                        )
                        logger.info("Added camera image to conversation")

                        if self.deps.camera_worker is not None:
                            np_img = self.deps.camera_worker.get_latest_frame()
                            if np_img is not None:
                                # Camera frames are BGR from OpenCV; convert so Gradio displays correct colors.
                                rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                            else:
                                rgb_frame = None
                            img = gr.Image(value=rgb_frame)

                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {
                                        "role": "assistant",
                                        "content": img,
                                    },
                                ),
                            )

                    # Let the robot reply out loud after tool calls
                    await self.connection.response.create(
                        response={
                            "instructions": "Use the tool result just returned and answer concisely in speech.",
                        },
                    )

                    # re synchronize the head wobble after a tool call that may have taken some time
                    if self.deps.head_wobbler is not None:
                        self.deps.head_wobbler.reset()

                # server error
                if event.type == "error":
                    err = getattr(event, "error", None)
                    msg = getattr(err, "message", str(err) if err else "unknown error")
                    code = getattr(err, "code", "")

                    logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

                    # Only show user-facing errors, not internal state errors
                    if code not in ("input_audio_buffer_commit_empty", "conversation_already_has_active_response"):
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                        )

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and process it.

        In full local mode, audio is processed entirely locally (VAD, ASR, LLM, TTS).
        Otherwise, audio is sent to the OpenAI server for processing.

        Handles both mono and stereo audio formats, converting to the expected
        mono format. Resamples if the input sample rate differs from the expected rate.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        # In local mode, we don't need an OpenAI connection
        if not self.connection and not self._is_full_local_mode:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Full local mode: use built-in VAD + ASR + LLM + TTS
        if self._is_full_local_mode:
            # Process with built-in VAD
            speech_started, speech_ended = self._local_vad.process(audio_frame)

            if speech_started:
                self._is_speech_active = True
                self._audio_buffer.clear()
                self.deps.movement_manager.set_listening(True)
                logger.info("VAD: speech started")

            if self._is_speech_active:
                self._audio_buffer.append(audio_frame.tobytes())

            if speech_ended and not self._vad_processing:
                self._vad_processing = True
                self._is_speech_active = False
                self.deps.movement_manager.set_listening(False)

                audio_data = b''.join(self._audio_buffer)
                self._audio_buffer.clear()
                logger.info("VAD: speech ended (%d bytes)", len(audio_data))

                # Process in background (ASR -> LLM -> TTS)
                asyncio.create_task(self._process_local_speech(audio_data))

            # Skip sending to OpenAI in full local mode
            return

        # Buffer audio for local ASR if speech is active (fallback when using OpenAI VAD)
        if (self._local_asr or self._local_asr_client) and self._is_speech_active:
            self._audio_buffer.append(audio_frame.tobytes())

        # Send to OpenAI (guard against races during reconnect)
        try:
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def _process_local_speech(self, audio_data: bytes) -> None:
        """Process speech audio: ASR -> LLM -> TTS.

        Args:
            audio_data: Raw PCM audio bytes from the speech buffer.
        """
        try:
            # Transcribe with local ASR
            transcript = await self._transcribe_with_local_asr(audio_data)
            if not transcript:
                logger.warning("ASR returned no transcription")
                return

            # Show transcription in UI
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

            # Generate response with local LLM
            if self._local_llm_client:
                await self._generate_local_response(transcript)
            else:
                logger.warning("Local LLM not available, cannot generate response")

        finally:
            self._vad_processing = False

    async def _process_with_local_vad(self, audio_data: bytes) -> None:
        """Process audio with local VAD check then ASR/LLM if turn complete."""
        try:
            is_complete = await self._check_turn_complete(audio_data)

            if is_complete:
                logger.info("Local VAD: turn complete, proceeding to ASR")
                self._is_speech_active = False
                self._vad_speech_frames = 0
                self._audio_buffer.clear()
                self.deps.movement_manager.set_listening(False)

                # Process with ASR and LLM (skip VAD check since we just did it)
                await self._process_local_asr(audio_data, check_vad=False)
            else:
                logger.info("Local VAD: turn incomplete, continuing to listen")
                # Keep listening - don't clear buffer, just reset silence counter
                self._vad_silence_frames = 0
        finally:
            self._vad_processing = False

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True
        # Cancel any pending debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Try to discover available voices for the configured realtime model.

        Attempts to retrieve model metadata from the OpenAI Models API and look
        for any keys that might contain voice names. Falls back to a curated
        list known to work with realtime if discovery fails.

        In full local mode with Chatterbox TTS, returns an empty list since
        voice selection is handled via reference audio, not voice names.
        """
        # In full local mode, Chatterbox uses reference audio instead of voice names
        if self._is_full_local_mode:
            return []

        # Conservative fallback list with default first
        fallback = [
            "cedar",
            "alloy",
            "aria",
            "ballad",
            "verse",
            "sage",
            "coral",
        ]

        # If client not initialized, return fallback
        if not hasattr(self, "client") or self.client is None:
            return fallback

        try:
            # Best effort discovery; safe-guarded for unexpected shapes
            model = await self.client.models.retrieve(config.MODEL_NAME)
            # Try common serialization paths
            raw = None
            for attr in ("model_dump", "to_dict"):
                fn = getattr(model, attr, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        pass
            if raw is None:
                try:
                    raw = dict(model)
                except Exception:
                    raw = None
            # Scan for voice candidates
            candidates: set[str] = set()

            def _collect(obj: object) -> None:
                try:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            kl = str(k).lower()
                            if "voice" in kl and isinstance(v, (list, tuple)):
                                for item in v:
                                    if isinstance(item, str):
                                        candidates.add(item)
                                    elif isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                                        candidates.add(item["name"])
                            else:
                                _collect(v)
                    elif isinstance(obj, (list, tuple)):
                        for it in obj:
                            _collect(it)
                except Exception:
                    pass

            if isinstance(raw, dict):
                _collect(raw)
            # Ensure default present and stable order
            voices = sorted(candidates) if candidates else fallback
            if "cedar" not in voices:
                voices = ["cedar", *[v for v in voices if v != "cedar"]]
            return voices
        except Exception:
            return fallback

    def _persist_api_key_if_needed(self) -> None:
        """Persist the API key into `.env` inside `instance_path/` when appropriate.

        - Only runs in Gradio mode when key came from the textbox and is non-empty.
        - Only saves if `self.instance_path` is not None.
        - Writes `.env` to `instance_path/.env` (does not overwrite if it already exists).
        - If `instance_path/.env.example` exists, copies its contents while overriding OPENAI_API_KEY.
        """
        try:
            if not self.gradio_mode:
                logger.warning("Not in Gradio mode; skipping API key persistence.")
                return

            if self._key_source != "textbox":
                logger.info("API key not provided via textbox; skipping persistence.")
                return

            key = (self._provided_api_key or "").strip()
            if not key:
                logger.warning("No API key provided via textbox; skipping persistence.")
                return
            if self.instance_path is None:
                logger.warning("Instance path is None; cannot persist API key.")
                return

            # Update the current process environment for downstream consumers
            try:
                import os

                os.environ["OPENAI_API_KEY"] = key
            except Exception:  # best-effort
                pass

            target_dir = Path(self.instance_path)
            env_path = target_dir / ".env"
            if env_path.exists():
                # Respect existing user configuration
                logger.info(".env already exists at %s; not overwriting.", env_path)
                return

            example_path = target_dir / ".env.example"
            content_lines: list[str] = []
            if example_path.exists():
                try:
                    content = example_path.read_text(encoding="utf-8")
                    content_lines = content.splitlines()
                except Exception as e:
                    logger.warning("Failed to read .env.example at %s: %s", example_path, e)

            # Replace or append the OPENAI_API_KEY line
            replaced = False
            for i, line in enumerate(content_lines):
                if line.strip().startswith("OPENAI_API_KEY="):
                    content_lines[i] = f"OPENAI_API_KEY={key}"
                    replaced = True
                    break
            if not replaced:
                content_lines.append(f"OPENAI_API_KEY={key}")

            # Ensure file ends with newline
            final_text = "\n".join(content_lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Created %s and stored OPENAI_API_KEY for future runs.", env_path)
        except Exception as e:
            # Never crash the app for QoL persistence; just log.
            logger.warning("Could not persist OPENAI_API_KEY to .env: %s", e)
