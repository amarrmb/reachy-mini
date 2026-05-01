"""Local audio processing: VAD, ASR, and TTS without external dependencies.

This module provides built-in local alternatives to cloud services:
- VAD: Energy-based voice activity detection
- ASR: Distil-Whisper for speech-to-text (lightweight, edge-optimized)
- TTS: Kokoro-82M for text-to-speech (lightweight, edge-optimized)
"""

import io
import re
import wave
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)

# Default sample rate for audio output
DEFAULT_SAMPLE_RATE = 24000


def clean_text_for_speech(text: str) -> str:
    """Clean text for TTS by removing special characters and formatting.

    Args:
        text: Raw text that may contain brackets, asterisks, etc.

    Returns:
        Cleaned text suitable for speech synthesis
    """
    # Strip emoji and other pictographs — Kokoro otherwise reads them as
    # words ("woman dancing", "hugging face", etc.). Covers BMP supplementary
    # symbols + emoticons + dingbats + flags + skin-tone modifiers.
    text = re.sub(
        r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF\u200d\ufe0f]',
        '', text)

    # Remove content in parentheses like (Pauses, then softly)
    text = re.sub(r'\([^)]*\)', '', text)

    # Remove content in square brackets like [thinking]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Remove content in curly braces like {stage direction}
    text = re.sub(r'\{[^}]*\}', '', text)

    # Remove asterisks used for emphasis like *italic* or **bold**
    text = re.sub(r'\*+', '', text)

    # Remove underscores used for emphasis
    text = re.sub(r'_+', '', text)

    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)

    # Trim leading/trailing whitespace
    text = text.strip()

    return text


class LocalVAD:
    """Energy-based Voice Activity Detection."""

    def __init__(
        self,
        energy_threshold: float = 0.01,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.3,
        sample_rate: int = 24000,
    ):
        """Initialize the VAD.

        Args:
            energy_threshold: RMS threshold for speech detection (0.0-1.0)
            silence_duration: Seconds of silence before considering speech ended
            min_speech_duration: Minimum seconds of speech to be valid
            sample_rate: Audio sample rate in Hz
        """
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.sample_rate = sample_rate

        # State
        self.is_speaking = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self._current_time = 0.0

    def reset(self) -> None:
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self._current_time = 0.0

    def process(self, audio_frame: np.ndarray) -> tuple[bool, bool]:
        """Process an audio frame and detect speech.

        Args:
            audio_frame: Audio samples as int16 numpy array

        Returns:
            Tuple of (speech_started, speech_ended)
        """
        # Calculate frame duration
        frame_duration = len(audio_frame) / self.sample_rate
        self._current_time += frame_duration

        # Calculate RMS energy
        audio_float = audio_frame.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))

        speech_started = False
        speech_ended = False

        if rms > self.energy_threshold:
            # Speech detected
            self.last_speech_time = self._current_time

            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = self._current_time
                speech_started = True
                logger.debug("VAD: Speech started (RMS=%.4f)", rms)

        elif self.is_speaking:
            # Silence detected while speaking
            silence_time = self._current_time - (self.last_speech_time or self._current_time)

            if silence_time >= self.silence_duration:
                # Check if speech was long enough
                speech_duration = (self.last_speech_time or self._current_time) - (self.speech_start_time or 0)

                if speech_duration >= self.min_speech_duration:
                    speech_ended = True
                    logger.debug("VAD: Speech ended (duration=%.2fs)", speech_duration)
                else:
                    logger.debug("VAD: Speech too short (%.2fs), ignoring", speech_duration)

                self.is_speaking = False
                self.speech_start_time = None

        return speech_started, speech_ended


class LocalASR:
    """Local Automatic Speech Recognition using distil-whisper (lightweight for edge devices)."""

    def __init__(
        self,
        model_name: str = "distil-whisper/distil-small.en",
        device: str = "auto",
        dtype: str = "auto",
        language: str = "en",
    ):
        """Initialize the ASR.

        Args:
            model_name: Distil-Whisper model (distil-small.en, distil-medium.en, distil-large-v3)
            device: Device to use (auto, cpu, cuda)
            dtype: Data type (auto, float16, float32)
            language: Language code for transcription
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.language = language
        self._model = None
        self._initialized = False
        import os as _os
        self._remote_endpoint = _os.environ.get("REMOTE_STT_ENDPOINT", "").strip() or None

    def _ensure_initialized(self) -> bool:
        """Lazy-load the distil-whisper model."""
        if self._initialized:
            return self._model is not None

        self._initialized = True

        if self._remote_endpoint:
            logger.info("LocalASR using REMOTE endpoint: %s (skipping distil-whisper load)", self._remote_endpoint)
            self._model = "remote"
            return True

        try:
            from distil_whisper_fastrtc import DistilWhisperSTT

            # Determine device
            device = self.device
            dtype = self.dtype

            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            if dtype == "auto":
                dtype = "float16" if device == "cuda" else "float32"

            logger.info("Loading Distil-Whisper model '%s' on %s with %s...",
                       self.model_name, device, dtype)

            self._model = DistilWhisperSTT(
                model=self.model_name,
                device=device,
                dtype=dtype,
            )

            logger.info("Distil-Whisper model loaded successfully")
            return True

        except ImportError:
            logger.error("distil-whisper-fastrtc not installed. Install with: pip install distil-whisper-fastrtc")
            return False
        except Exception as e:
            logger.error("Failed to load Distil-Whisper model: %s", e)
            return False

    async def transcribe(self, audio_data: bytes, sample_rate: int = 24000) -> Optional[str]:
        """Transcribe audio to text.

        Args:
            audio_data: Raw PCM audio bytes (16-bit signed)
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text or None if failed
        """
        if not self._ensure_initialized():
            return None

        try:
            # Convert bytes to numpy array (int16)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Remote path: POST WAV to jetson-assistant /stt/transcribe.
            if self._remote_endpoint:
                return await self._remote_transcribe(sample_rate, audio_array)

            # Run transcription in executor to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._transcribe_array, sample_rate, audio_array
            )

            return result

        except Exception as e:
            logger.error("ASR transcription failed: %s", e)
            return None


    async def _remote_transcribe(self, sample_rate: int, audio_array) -> Optional[str]:
        """POST a WAV to jetson-assistant /stt/transcribe and return its text."""
        import io, wave
        import aiohttp

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio_array.tobytes())
        wav_bytes = buf.getvalue()

        try:
            timeout = aiohttp.ClientTimeout(total=8.0)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                form = aiohttp.FormData()
                form.add_field("audio", wav_bytes, filename="audio.wav", content_type="audio/wav")
                async with sess.post(self._remote_endpoint, data=form) as resp:
                    if resp.status != 200:
                        logger.error("Remote STT %s -> %d", self._remote_endpoint, resp.status)
                        return None
                    data = await resp.json()
                    if not data.get("success"):
                        logger.warning("Remote STT failed: %s", data.get("error"))
                        return None
                    text = (data.get("text") or "").strip()
                    if text:
                        logger.info("Remote ASR transcription: %s", text[:120])
                    return text or None
        except Exception as e:
            logger.error("Remote STT call failed: %s", e)
            return None

    def _transcribe_array(self, sample_rate: int, audio_array: np.ndarray) -> Optional[str]:
        """Transcribe an audio array (runs in executor)."""
        try:
            # Call distil-whisper's stt method with (sample_rate, audio_data) tuple
            text = self._model.stt((sample_rate, audio_array))

            if text and text.strip():
                text = text.strip()
                logger.info("ASR transcription: %s", text[:100])
                return text

            return None

        except Exception as e:
            logger.error("Distil-Whisper transcription error: %s", e)
            return None


class LocalTTS:
    """Local Text-to-Speech using Kokoro via FastRTC.

    Uses FastRTC's built-in Kokoro support - a lightweight 82M parameter TTS
    optimized for edge devices with high-quality voice synthesis.
    """

    def __init__(
        self,
        output_sample_rate: int = 24000,
        voice: str = "af_sarah",
        speed: float = 1.0,
    ):
        """Initialize the TTS.

        Args:
            output_sample_rate: Target sample rate for output audio
            voice: Voice to use (af_sarah, am_michael, bf_emma, etc.)
            speed: Speech speed multiplier (0.5-2.0)
        """
        self.output_sample_rate = output_sample_rate
        self.voice = voice
        self.speed = speed
        self._model = None
        self._initialized = False
        import os as _os
        self._remote_endpoint = _os.environ.get("REMOTE_TTS_ENDPOINT", "").strip() or None

    def _ensure_initialized(self) -> bool:
        """Lazy-load the Kokoro TTS model using FastRTC."""
        if self._initialized:
            return self._model is not None

        self._initialized = True

        if self._remote_endpoint:
            logger.info("LocalTTS using REMOTE endpoint: %s (skipping Kokoro load)", self._remote_endpoint)
            self._model = "remote"
            return True

        try:
            from fastrtc import get_tts_model

            logger.info("Loading Kokoro TTS model (voice: %s)...", self.voice)

            # Use FastRTC's built-in Kokoro support
            self._model = get_tts_model(model="kokoro", voice=self.voice)

            logger.info("Kokoro TTS model loaded successfully")
            return True

        except ImportError:
            logger.error("FastRTC not installed properly. Install with: pip install fastrtc")
            return False
        except Exception as e:
            logger.error("Failed to load Kokoro TTS model: %s", e)
            return False

    async def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio samples as int16 numpy array, or None if failed
        """
        if not text or not text.strip():
            return None

        # Clean text for speech (remove brackets, asterisks, stage directions, etc.)
        cleaned_text = clean_text_for_speech(text)

        if not cleaned_text or not cleaned_text.strip():
            logger.warning("Text became empty after cleaning: %s", text[:100])
            return None

        if not self._ensure_initialized():
            return None

        # Remote path: POST {text,voice} to jetson-assistant /tts/synthesize.
        if self._remote_endpoint:
            return await self._remote_synthesize(cleaned_text)

        try:
            # Run synthesis in executor to not block
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, self._synthesize_sync, cleaned_text)
            return audio

        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            return None


    async def _remote_synthesize(self, text: str):
        """POST text to jetson-assistant /tts/synthesize, return int16 mono samples.

        Resamples to self.output_sample_rate if the server SR differs.
        """
        import io, wave
        import aiohttp
        import numpy as _np
        try:
            timeout = aiohttp.ClientTimeout(total=15.0)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                payload = {"text": text, "voice": self.voice}
                async with sess.post(self._remote_endpoint, json=payload) as resp:
                    if resp.status != 200:
                        body = (await resp.text())[:300]
                        logger.error("Remote TTS %s -> %d %s", self._remote_endpoint, resp.status, body)
                        return None
                    wav_bytes = await resp.read()
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                sr = w.getframerate()
                ch = w.getnchannels()
                sw = w.getsampwidth()
                frames = w.readframes(w.getnframes())
            if sw != 2:
                logger.warning("Remote TTS unexpected sample width %d", sw)
                return None
            audio = _np.frombuffer(frames, dtype=_np.int16)
            if ch == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(_np.int16)
            if sr != self.output_sample_rate:
                from scipy.signal import resample
                n = int(round(len(audio) * self.output_sample_rate / sr))
                audio = resample(audio, n).astype(_np.int16)
            logger.info("Remote TTS: %d samples @ %d Hz (%.2fs)", len(audio), self.output_sample_rate, len(audio)/self.output_sample_rate)
            return audio
        except Exception as e:
            logger.error("Remote TTS call failed: %s", e)
            return None

    def _synthesize_sync(self, text: str) -> Optional[np.ndarray]:
        """Synchronous synthesis (runs in executor)."""
        try:
            from scipy.signal import resample

            # Collect all audio chunks from the streaming TTS
            audio_chunks = []
            for audio_chunk in self._model.stream_tts_sync(text):
                audio_chunks.append(audio_chunk)

            if not audio_chunks:
                logger.warning("TTS produced no audio")
                return None

            # Concatenate all chunks
            # Assuming audio_chunks are (sample_rate, audio_data) tuples
            if isinstance(audio_chunks[0], tuple):
                # Extract audio data from tuples
                audio_arrays = [chunk[1] if isinstance(chunk, tuple) else chunk for chunk in audio_chunks]
                audio = np.concatenate(audio_arrays)
            else:
                audio = np.concatenate(audio_chunks)

            # Ensure int16 format
            if audio.dtype != np.int16:
                # Assume float32 in [-1, 1] range
                if audio.dtype in (np.float32, np.float64):
                    max_val = np.abs(audio).max()
                    if max_val > 1.0:
                        audio = audio / max_val
                    audio = (audio * 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)

            # Kokoro outputs at 24kHz by default
            model_sample_rate = 24000

            # Resample if needed
            if model_sample_rate != self.output_sample_rate:
                num_samples = int(len(audio) * self.output_sample_rate / model_sample_rate)
                audio = resample(audio, num_samples).astype(np.int16)

            # Apply speed adjustment if needed
            if self.speed != 1.0:
                num_samples = int(len(audio) / self.speed)
                audio = resample(audio, num_samples).astype(np.int16)

            logger.debug("TTS generated %d samples", len(audio))
            return audio

        except Exception as e:
            logger.error("Kokoro TTS synthesis error: %s", e)
            return None


# Convenience function to check local audio capabilities
def check_local_audio_support() -> dict[str, bool]:
    """Check which local audio components are available.

    Returns:
        Dict with availability status for each component
    """
    support = {
        "vad": True,  # Built-in, always available
        "asr_distil_whisper": False,
        "tts_kokoro": False,
    }

    try:
        from distil_whisper_fastrtc import DistilWhisperSTT
        support["asr_distil_whisper"] = True
    except ImportError:
        pass

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        support["tts_kokoro"] = True
    except ImportError:
        pass

    return support
