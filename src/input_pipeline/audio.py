"""
Audio Channel — Microphone capture with speech transcription, novelty, and prosody.

Captures a short audio clip when the heartbeat fires, transcribes it
using OpenAI Whisper, and computes novelty vs recent transcripts.

PROSODY: After capture, analyzes the audio signal for emotional intensity:
  - RMS energy variance (loud = animated)
  - Pitch variance (monotone = bored, varied = excited)
  - Speech rate (fast = energized, slow = calm)
  
High emotional intensity boosts the channel weight so creative interjections
land when the user is "in the flow" rather than during a lull.

Does NOT stream continuously — only listens when the engine is "paying attention."
"""

from __future__ import annotations

import io
import os
from collections import deque
from dataclasses import dataclass

import numpy as np

from src.models import ChannelInput


@dataclass
class ProsodyFeatures:
    """Extracted emotion/energy features from audio signal."""
    rms_energy: float = 0.0         # overall loudness (0-1 normalized)
    energy_variance: float = 0.0    # how much loudness fluctuates (animated speech)
    pitch_variance: float = 0.0     # monotone (0) vs melodic (1)
    speech_rate: float = 0.0        # syllables-per-second estimate (0-1 normalized)
    intensity: float = 0.0          # composite emotion score (0=flat, 1=very excited)

    @property
    def label(self) -> str:
        if self.intensity > 0.7:
            return "🔥 EXCITED"
        elif self.intensity > 0.4:
            return "⚡ ENGAGED"
        elif self.intensity > 0.2:
            return "😐 NEUTRAL"
        else:
            return "😴 CALM"


class AudioChannel:
    def __init__(
        self,
        history_window: int = 10,
        base_weight_direct: float = 1.0,
        base_weight_overheard: float = 0.25,
        capture_seconds: float = 2.0,
        sample_rate: int = 16000,
        api_key: str = "",
        device_index: int | None = None,
        vad_threshold: float = 0.003,
    ):
        self.history_window = history_window
        self.base_weight_direct = base_weight_direct
        self.base_weight_overheard = base_weight_overheard
        self.capture_seconds = capture_seconds
        self.sample_rate = sample_rate
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.device_index = device_index
        self.vad_threshold = vad_threshold
        self._transcript_history: deque[str] = deque(maxlen=history_window)
        self._available = False
        self._initialized = False
        self._whisper_client = None
        self.paused = False
        self._last_prosody: ProsodyFeatures = ProsodyFeatures()

    @staticmethod
    def list_devices() -> list[dict]:
        """List all available audio input devices. Returns list of {index, name, channels, sample_rate}."""
        devices = []
        try:
            import sounddevice as sd
        except ImportError:
            return devices
        all_devices = sd.query_devices()
        for i, dev in enumerate(all_devices):
            if dev["max_input_channels"] > 0:
                devices.append({
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": int(dev["default_samplerate"]),
                    "is_default": dev == sd.query_devices(kind="input"),
                })
        return devices

    def initialize(self) -> bool:
        """Check if audio capture is possible."""
        if self._initialized:
            return self._available
        self._initialized = True
        try:
            import sounddevice as sd
            if self.device_index is not None:
                device_info = sd.query_devices(self.device_index)
                if device_info["max_input_channels"] > 0:
                    sd.default.device[0] = self.device_index
                    self._available = True
                    print(f"   [Audio] Microphone: connected (device {self.device_index}: {device_info['name'][:40]})")
                else:
                    print(f"   [Audio] Device {self.device_index} ({device_info['name']}) is not an input device")
                    self._available = False
            else:
                default_input = sd.query_devices(kind="input")
                if default_input is not None:
                    self._available = True
                    print(f"   [Audio] Microphone: connected ({default_input['name'][:40]})")
                else:
                    print("   [Audio] Microphone: no input device found")
                    self._available = False
        except ImportError:
            print("   [Audio] Microphone: sounddevice not installed")
            self._available = False
        except Exception as e:
            print(f"   [Audio] Microphone error: {e}")
            self._available = False
        return self._available

    @property
    def is_available(self) -> bool:
        return self._available

    def capture_audio(self, quiet: bool = False) -> np.ndarray | None:
        """Record a short fixed-duration audio clip. Used by heartbeat perception."""
        if not self._available:
            return None
        try:
            import sounddevice as sd
            if not quiet:
                print(f"   [MIC ON]  Listening for {self.capture_seconds:.0f} seconds -- talk now!")
            audio = sd.rec(
                int(self.capture_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            if not quiet:
                print(f"   [MIC OFF] Recording done.")
            return audio.flatten()
        except Exception as e:
            if not quiet:
                print(f"   [MIC OFF] Capture error: {e}")
            return None

    def capture_smart(
        self,
        probe_seconds: float = 0.5,
        max_seconds: float = 10.0,
        silence_timeout: float = 1.2,
        quiet: bool = True,
        debug_rms: bool = False,
    ) -> np.ndarray | None:
        """Voice-activity-driven capture: waits for speech, records until
        the speaker pauses, then returns the full utterance as one array.

        probe_seconds  — length of each mini-recording chunk
        max_seconds    — hard cap on total recording length
        silence_timeout — stop after this many seconds of silence following speech
        debug_rms      — print RMS levels for threshold tuning
        """
        if not self._available:
            return None
        try:
            import sounddevice as sd

            chunk_samples = int(probe_seconds * self.sample_rate)
            chunks: list[np.ndarray] = []
            speech_started = False
            silent_since = 0.0
            total_seconds = 0.0

            while total_seconds < max_seconds:
                if self.paused:
                    if chunks:
                        break
                    return None

                chunk = sd.rec(chunk_samples, samplerate=self.sample_rate,
                               channels=1, dtype="float32")
                sd.wait()
                flat = chunk.flatten()
                total_seconds += probe_seconds

                rms = float(np.sqrt(np.mean(flat.astype(np.float64) ** 2)))
                is_speech = rms > self.vad_threshold

                if debug_rms:
                    bar = "#" * min(int(rms * 2000), 50)
                    marker = " << SPEECH" if is_speech else ""
                    print(f"   [RMS] {rms:.5f} [{bar}]{marker}")

                if is_speech:
                    if not speech_started:
                        speech_started = True
                        self._play_listening_tone()
                        if not quiet:
                            print(f"   * Recording...")
                    chunks.append(flat)
                    silent_since = 0.0
                elif speech_started:
                    chunks.append(flat)
                    silent_since += probe_seconds
                    if silent_since >= silence_timeout:
                        break

            if not chunks:
                return None

            return np.concatenate(chunks)

        except Exception as e:
            if not quiet:
                print(f"   [Audio] Smart capture error: {e}")
            return None

    def _play_listening_tone(self) -> None:
        """Play a short audible tone so the user knows recording has started.
        Uses winsound on Windows (separate from sounddevice) to avoid
        conflicts with ongoing audio recording."""
        try:
            import sys
            if sys.platform == "win32":
                import winsound
                winsound.Beep(880, 120)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass

    def has_speech(self, audio: np.ndarray, threshold: float | None = None) -> bool:
        """Voice activity detection — is the RMS energy above threshold?"""
        audio_f = audio.astype(np.float64)
        rms = np.sqrt(np.mean(audio_f ** 2))
        t = threshold if threshold is not None else self.vad_threshold
        return float(rms) > t

    def _get_whisper_client(self):
        """Reuse a single AsyncOpenAI client for all transcriptions."""
        if self._whisper_client is None:
            from openai import AsyncOpenAI
            self._whisper_client = AsyncOpenAI(api_key=self._api_key or None)
        return self._whisper_client

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using OpenAI Whisper API. Uses in-memory buffer."""
        try:
            import wave

            buf = io.BytesIO()
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())

            buf.seek(0)
            buf.name = "audio.wav"

            client = self._get_whisper_client()
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=buf,
                language="en",
            )
            return response.text.strip()

        except Exception as e:
            print(f"   [Audio] Transcription error: {e}")
            return ""

    def compute_novelty(self, transcript: str) -> float:
        """Compare transcript to recent history. New topics = high novelty."""
        if not transcript:
            return 0.0

        if not self._transcript_history:
            self._transcript_history.append(transcript)
            return 0.5

        current_words = set(transcript.lower().split())
        if len(current_words) < 2:
            return 0.3

        similarities = []
        for past in self._transcript_history:
            past_words = set(past.lower().split())
            if not past_words:
                continue
            overlap = len(current_words & past_words)
            union = len(current_words | past_words)
            similarities.append(overlap / union if union > 0 else 0.0)

        self._transcript_history.append(transcript)

        if not similarities:
            return 1.0

        max_similarity = max(similarities)
        return max(0.0, min(1.0, 1.0 - max_similarity))

    def analyze_prosody(self, audio: np.ndarray) -> ProsodyFeatures:
        """Extract emotion/prosody features from raw audio signal.

        Uses signal-level heuristics that work without ML models:
        - RMS energy and its frame-to-frame variance
        - Zero-crossing rate as a pitch proxy
        - Segment density for speech rate estimation
        """
        if audio is None or len(audio) < self.sample_rate * 0.3:
            return ProsodyFeatures()

        audio_f = audio.astype(np.float64)

        # --- RMS energy (overall and variance across 50ms frames) ---
        frame_size = int(self.sample_rate * 0.05)
        n_frames = len(audio_f) // frame_size
        if n_frames < 2:
            return ProsodyFeatures()

        frames = audio_f[:n_frames * frame_size].reshape(n_frames, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

        rms_energy = float(np.mean(frame_rms))
        energy_variance = float(np.std(frame_rms))
        rms_normalized = min(1.0, rms_energy / 0.08)
        energy_var_normalized = min(1.0, energy_variance / 0.04)

        # --- Pitch proxy via zero-crossing rate variance ---
        signs = np.sign(audio_f)
        zero_crossings = np.abs(np.diff(signs)) / 2
        zc_frames = zero_crossings[:n_frames * frame_size - 1].reshape(n_frames, frame_size - 1)
        zcr_per_frame = np.sum(zc_frames, axis=1) / frame_size
        pitch_variance = float(np.std(zcr_per_frame))
        pitch_var_normalized = min(1.0, pitch_variance / 0.05)

        # --- Speech rate proxy: voiced segment density ---
        speech_frames = frame_rms > self.vad_threshold
        speech_ratio = float(np.mean(speech_frames))
        transitions = np.sum(np.abs(np.diff(speech_frames.astype(int))))
        duration = len(audio_f) / self.sample_rate
        segments_per_second = transitions / (2 * duration) if duration > 0 else 0
        speech_rate_normalized = min(1.0, segments_per_second / 4.0)

        # --- Composite intensity ---
        intensity = (
            0.25 * rms_normalized
            + 0.30 * energy_var_normalized
            + 0.25 * pitch_var_normalized
            + 0.20 * speech_rate_normalized
        )
        intensity = max(0.0, min(1.0, intensity))

        return ProsodyFeatures(
            rms_energy=rms_normalized,
            energy_variance=energy_var_normalized,
            pitch_variance=pitch_var_normalized,
            speech_rate=speech_rate_normalized,
            intensity=intensity,
        )

    async def quick_capture_and_transcribe(self) -> str:
        """
        Capture audio and transcribe in one step. Used by the background listener.
        Returns the transcript string, or empty string if silence/failure.
        """
        if not self._available:
            return ""

        audio = self.capture_audio()
        if audio is None:
            return ""

        if not self.has_speech(audio):
            return ""

        return await self.transcribe(audio)

    async def process(self) -> ChannelInput:
        """
        Full audio pipeline: capture → detect speech → transcribe → novelty → prosody.
        Prosody intensity boosts effective_weight so the engine pays more attention
        when the user sounds excited/animated.
        """
        if not self._available:
            return ChannelInput(
                channel="audio",
                raw_content="",
                novelty=0.0,
                base_weight=0.0,
                effective_weight=0.0,
                available=False,
            )

        audio = self.capture_audio()
        if audio is None:
            return ChannelInput(
                channel="audio",
                raw_content="[capture failed]",
                novelty=0.0,
                base_weight=0.0,
                effective_weight=0.0,
                available=False,
            )

        if not self.has_speech(audio):
            print("   [Audio] Audio: silence detected")
            return ChannelInput(
                channel="audio",
                raw_content="[silence]",
                novelty=0.0,
                base_weight=0.0,
                effective_weight=0.0,
                available=True,
            )

        print("   [Audio] Audio: speech detected, transcribing...")
        transcript = await self.transcribe(audio)

        if not transcript:
            return ChannelInput(
                channel="audio",
                raw_content="[transcription empty]",
                novelty=0.0,
                base_weight=self.base_weight_overheard,
                effective_weight=0.0,
                available=True,
            )

        novelty = self.compute_novelty(transcript)
        prosody = self.analyze_prosody(audio)
        self._last_prosody = prosody

        # Boost weight when user sounds excited: up to 1.5x at max intensity
        emotion_boost = 1.0 + 0.5 * prosody.intensity
        base_weight = self.base_weight_overheard
        effective_weight = base_weight * novelty * emotion_boost

        novelty_label = "🔴 HIGH" if novelty > 0.6 else "🟡 MED" if novelty > 0.3 else "⚪ LOW"
        print(f"   [Audio] Audio [{novelty_label} novelty={novelty:.2f}]: \"{transcript[:80]}\"")
        print(f"   [Audio] Prosody: {prosody.label} (intensity={prosody.intensity:.2f}, boost={emotion_boost:.2f}x)")

        return ChannelInput(
            channel="audio",
            raw_content=transcript,
            novelty=novelty,
            base_weight=base_weight,
            effective_weight=effective_weight,
            available=True,
        )
