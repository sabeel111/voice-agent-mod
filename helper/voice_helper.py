#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "onnx-asr[cpu,hub]>=0.12",
#   "moonshine-voice>=0.1.5",
#   "ptts>=0.4",
#   "webrtcvad-wheels>=2.0.14",
# ]
# ///
"""Local voice I/O for the voice-agent Claude Code mod.

Everything runs on this machine: the microphone is read through PipeWire /
PulseAudio / ALSA command-line tools, speech is detected with WebRTC VAD,
transcribed with Moonshine (moonshine-voice) or NVIDIA Parakeet (onnx-asr; on
the GPU when there is an NVIDIA one) and spoken with Phonon (ptts). Pipecat's
Smart Turn model decides when the person has finished speaking.

The mod starts this process and reads newline-delimited JSON events from its
stdout. It sends commands over HTTP on a Unix socket (``--socket``), which is
also what a Hyprland push-to-talk key binding talks to:

    POST /speak     {"text": "...", "interrupt": false}
    POST /stop                              stop speaking, drop the queue
    POST /mode      {"mode": "natural" | "wake" | "ptt"}
    POST /mute      {"muted": true}
    POST /ptt/down, /ptt/up, /ptt/toggle    push-to-talk
    POST /ask       {"question": "...", "timeout": 45}   -> {"answer": "yes"|"no"|"timeout"|...}
    GET  /status
    POST /shutdown

Events on stdout: ready, state, transcript, barge_in, wake, log, error.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import os
import queue
import re
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, fields
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

MIC_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = MIC_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2

MODES = ("natural", "wake", "ptt")

DEFAULT_STT_MODEL = "moonshine-medium"
PARAKEET_DEFAULT = "nemo-parakeet-tdt-0.6b-v2"
# onnx-asr model name -> Hugging Face repo with its ONNX export.
STT_MODELS = {
    "nemo-parakeet-tdt-0.6b-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",  # English
    "nemo-parakeet-tdt-0.6b-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",  # 25 European languages
}
# Moonshine streaming models (MIT), largest first: setting name -> model arch.
MOONSHINE_MODELS = {
    "moonshine-medium": "medium-streaming",  # 245M, English
    "moonshine-small": "small-streaming",  # 123M, also es, de, ja
    "moonshine-tiny": "tiny-streaming",  # 34M, the most languages
}
STT_ALIASES = {"parakeet": PARAKEET_DEFAULT, "parakeet-v2": PARAKEET_DEFAULT,
               "parakeet-v3": "nemo-parakeet-tdt-0.6b-v3", "moonshine": DEFAULT_STT_MODEL}
GPU_RUNTIME = "onnxruntime-gpu[cuda,cudnn]>=1.20"

YES_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "allow", "approve",
    "approved", "fine", "alright", "affirmative", "correct", "absolutely",
    "definitely", "go", "proceed", "continue", "accept",
}
YES_PHRASES = ("go ahead", "do it", "please do", "of course", "sounds good", "let it", "that's fine")
NO_WORDS = {
    "no", "nope", "nah", "deny", "denied", "stop", "cancel", "don't", "dont",
    "wait", "negative", "never", "reject", "refuse", "block",
}
NO_PHRASES = ("do not", "not now", "hold on", "hang on", "not yet", "no way")

# Things a recognizer may "hear" in silence or noise.
HALLUCINATIONS = {
    "", "you", "thank you", "thanks", "thank you.", "bye", "bye.", "okay.",
    "thanks for watching", "thank you for watching", "subtitles by the amara.org community",
}


# --------------------------------------------------------------------------
# Configuration


@dataclass
class Config:
    mode: str = "wake"
    wake_word: str = "hey claude"
    wake_aliases: list[str] = field(default_factory=lambda: ["hey cloud", "hey clyde", "hi claude", "okay claude"])
    wake_threshold: float = 0.75
    follow_up_s: float = 8.0
    stt_model: str = DEFAULT_STT_MODEL
    stt_language: str = "en"
    stt_device: str = "auto"  # auto | cuda | cpu (Parakeet only)
    # Names and jargon the recognizer should favour (Moonshine only); the
    # wake word's name ("Claude") is always among them.
    stt_keyterms: list[str] = field(default_factory=list)
    stt_quant: str = ""  # "" picks int8 on the CPU, full precision on the GPU
    tts_model: str = ""
    tts_voice: str = ""
    tts_lang: str = "en"
    tts_quant: str = "q8"
    barge_in: bool = True
    barge_in_rms: float = 0.03
    barge_in_ms: int = 300
    # Hands-free speech must be this many times louder than the background
    # noise floor (a slow running average of the mic level), so other voices
    # and the TV don't count as speech. 0 or less turns the check off.
    loudness_ratio: float = 2.0
    # Without turn detection (or when its model fails to load), an utterance
    # ends after this much silence.
    silence_ms: int = 800
    # Transcribe this far into a pause, so the text is ready when the pause
    # turns out to be the end of the utterance. The turn model also runs here.
    early_ms: int = 150
    # Smart Turn: an audio model judges, early into each pause, whether the
    # person has finished. Finished: answer at once. Not finished: wait for up
    # to turn_max_silence_ms, so a pause mid-sentence does not cut them off.
    turn_detection: bool = True
    turn_threshold: float = 0.5  # probability of "finished" that ends the turn
    turn_max_silence_ms: int = 1200
    min_utterance_ms: int = 500
    max_utterance_s: float = 30.0
    vad_aggressiveness: int = 3
    capture_cmd: list[str] = field(default_factory=list)
    playback_cmd: list[str] = field(default_factory=list)
    chimes: bool = True

    @classmethod
    def from_json(cls, raw: str | None) -> "Config":
        cfg = cls()
        if not raw:
            return cfg
        data = json.loads(raw)
        names = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in names and value is not None and value != "":
                setattr(cfg, key, value)
        if cfg.mode not in MODES:
            cfg.mode = "wake"
        return cfg


# --------------------------------------------------------------------------
# Events to the mod


_emit_lock = threading.Lock()


def emit(kind: str, **payload) -> None:
    line = json.dumps({"type": kind, **payload}, ensure_ascii=False)
    with _emit_lock:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, ValueError):
            # The mod is gone; nobody is listening to us any more.
            os._exit(0)


def log(msg: str, level: str = "info") -> None:
    emit("log", level=level, msg=msg)


# --------------------------------------------------------------------------
# Text helpers (pure, unit-tested)


def normalize_words(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w for w in text.split() if w]


def match_wake(text: str, phrases: Iterable[str], threshold: float) -> Optional[str]:
    """Return the words after the wake phrase, or None when there is none.

    The phrase may be preceded by a filler word or two ("um, hey Claude") and
    is matched fuzzily, since Whisper writes "Hey Claude" as "hey cloud" or
    "hey, Clyde" often enough.
    """
    words = normalize_words(text)
    if not words:
        return None
    best: Optional[tuple[float, int]] = None
    for phrase in phrases:
        target = normalize_words(phrase)
        if not target:
            continue
        n = len(target)
        for start in range(0, min(3, max(1, len(words) - n + 1))):
            window = words[start:start + n]
            if len(window) < n:
                continue
            ratio = difflib.SequenceMatcher(None, " ".join(window), " ".join(target)).ratio()
            if ratio >= threshold and (best is None or ratio > best[0]):
                best = (ratio, start + n)
    if best is None:
        return None
    # Keep the original casing and punctuation of what follows the phrase.
    rest_words = words[best[1]:]
    if not rest_words:
        return ""
    return _tail_from(text, rest_words)


def _tail_from(text: str, rest_words: list[str]) -> str:
    """Find the original-text tail that starts at rest_words[0]."""
    lowered = text.lower().replace("’", "'")
    first = rest_words[0]
    # Count how many times the first remaining word occurs before, in words.
    all_words = normalize_words(text)
    occurrence = len(all_words) - len(rest_words)
    idx = -1
    seen = 0
    for m in re.finditer(r"[a-z0-9']+", lowered):
        if seen == occurrence:
            idx = m.start()
            break
        seen += 1
    if idx < 0 or not lowered[idx:].startswith(first):
        return " ".join(rest_words)
    return text[idx:].strip(" ,.;:!?-")


def classify_yes_no(text: str) -> str:
    words = normalize_words(text)
    joined = " ".join(words)
    if not words:
        return "unclear"
    if any(p in joined for p in NO_PHRASES) or any(w in NO_WORDS for w in words):
        return "no"
    if any(p in joined for p in YES_PHRASES) or any(w in YES_WORDS for w in words):
        return "yes"
    return "unclear"


def is_echo(heard: str, spoken: str, threshold: float = 0.6) -> bool:
    """True when what the mic heard is mostly what we just said ourselves."""
    a = normalize_words(heard)
    b = normalize_words(spoken)
    if not a or not b:
        return False
    if len(a) <= 2:
        return " ".join(a) in " ".join(b)
    common = sum(1 for w in a if w in set(b))
    return common / len(a) >= threshold


# Words a finished request almost never ends on: a pause after one of them is
# a pause for thought, whatever the voice sounds like.
DANGLING_WORDS = {
    "a", "an", "the", "to", "of", "for", "with", "from", "into", "onto", "about",
    "and", "or", "but", "nor", "because", "whether", "if", "than", "that", "so",
    "my", "your", "our", "their", "its", "um", "uh", "erm", "er", "uhm", "hmm",
}


def ends_mid_sentence(text: str) -> bool:
    """True when a (partial) transcript clearly stops in the middle of a thought."""
    t = text.strip()
    if not t:
        return False
    if t.endswith((",", "...", "…", "-", "—", ";", ":")):
        return True
    words = normalize_words(t)
    return bool(words) and words[-1] in DANGLING_WORDS


def is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    return t in HALLUCINATIONS or not re.search(r"[a-z0-9]", t)


def rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


def pcm16(chunk: np.ndarray) -> bytes:
    return (np.clip(chunk, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def tone(freqs: list[float], rate: int, ms: int = 90, gain: float = 0.18) -> bytes:
    out = []
    for f in freqs:
        t = np.arange(int(rate * ms / 1000)) / rate
        env = np.minimum(1.0, np.minimum(t, t[::-1]) * 60)  # ~16 ms fades
        out.append(np.sin(2 * math.pi * f * t) * env * gain)
    return pcm16(np.concatenate(out).astype(np.float32))


# --------------------------------------------------------------------------
# Audio devices


def _first_available(candidates: list[list[str]]) -> Optional[list[str]]:
    for argv in candidates:
        if shutil.which(argv[0]):
            return argv
    return None


def capture_command(cfg: Config) -> list[str]:
    if cfg.capture_cmd:
        return list(cfg.capture_cmd)
    argv = _first_available([
        ["parecord", "--raw", "--format=s16le", f"--rate={MIC_RATE}", "--channels=1", "--latency-msec=30"],
        ["pw-record", "--format", "s16", "--rate", str(MIC_RATE), "--channels", "1", "--raw", "-"],
        ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(MIC_RATE), "-c", "1"],
    ])
    if argv is None:
        raise RuntimeError("no recorder found: install pipewire-pulse (parecord) or alsa-utils (arecord)")
    return argv


def playback_command(cfg: Config, rate: int) -> list[str]:
    if cfg.playback_cmd:
        return [a.replace("{rate}", str(rate)) for a in cfg.playback_cmd]
    argv = _first_available([
        ["paplay", "--raw", "--format=s16le", f"--rate={rate}", "--channels=1", "--latency-msec=40"],
        ["pw-play", "--format", "s16", "--rate", str(rate), "--channels", "1", "--raw", "-"],
        ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(rate), "-c", "1"],
    ])
    if argv is None:
        raise RuntimeError("no player found: install pipewire-pulse (paplay) or alsa-utils (aplay)")
    return argv


class Player:
    """Plays raw 16-bit mono PCM through a player process; stop() is instant."""

    def __init__(self, argv: list[str]):
        self.argv = argv
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def write(self, data: bytes) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc = self._proc
        try:
            assert proc.stdin is not None
            proc.stdin.write(data)
            proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def drain(self) -> None:
        """Close stdin and wait until the player has played everything."""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=60)
        except Exception:
            proc.kill()

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass


def play_once(argv: list[str], data: bytes) -> None:
    def run():
        try:
            subprocess.run(argv, input=data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


# --------------------------------------------------------------------------
# Engines (loaded lazily so tests can swap them)


def resolve_stt_model(name: str) -> str:
    name = (name or "").strip()
    name = STT_ALIASES.get(name, name)
    if name not in STT_MODELS and name not in MOONSHINE_MODELS:
        if name:
            log(f"unknown speech recognition model {name!r}; using {DEFAULT_STT_MODEL}", "warn")
        return DEFAULT_STT_MODEL
    return name


def is_parakeet(name: str) -> bool:
    return name in STT_MODELS


def stt_keyterms(cfg: Config) -> list[str]:
    """The words to bias recognition towards: the wake word's name, then the person's own."""
    terms = [w.capitalize() for w in normalize_words(cfg.wake_word) if w not in ("hey", "hi", "ok", "okay")]
    terms += [" ".join(str(t).replace(",", " ").split()) for t in cfg.stt_keyterms]
    out: list[str] = []
    for t in terms:
        if t and t.lower() not in (o.lower() for o in out):
            out.append(t)
    return out


def make_stt(cfg: Config):
    name = resolve_stt_model(cfg.stt_model)
    return ParakeetSTT(cfg) if is_parakeet(name) else MoonshineSTT(cfg, name)


def cuda_available() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


class ParakeetSTT:
    """NVIDIA Parakeet TDT through onnx-asr: fast, and quiet on silence."""

    def __init__(self, cfg: Config):
        import onnx_asr
        from huggingface_hub import snapshot_download

        self.cfg = cfg
        name = resolve_stt_model(cfg.stt_model)
        self.name = name = name if is_parakeet(name) else PARAKEET_DEFAULT
        # A plain folder rather than the hub cache: newer onnxruntime refuses
        # external weight files that sit behind the cache's symlinks.
        folder = os.path.join(os.path.expanduser("~/.cache/voice-agent/models"), name)

        def load(quant: str, providers: list[str]):
            stem = f"*.{quant}.onnx" if quant else "*-model.onnx*"
            snapshot_download(STT_MODELS[name], local_dir=folder,
                              allow_patterns=[stem, "nemo*.onnx", "vocab.txt", "config.json"])
            return onnx_asr.load_model(name, folder, quantization=quant or None, providers=providers)

        self.model = None
        if cfg.stt_device != "cpu" and cuda_available():
            try:
                import onnxruntime as ort

                ort.preload_dlls()
                # Full precision: int8 kernels mostly run on the CPU anyway.
                self.model = load(cfg.stt_quant, ["CUDAExecutionProvider", "CPUExecutionProvider"])
                self.device = "gpu"
            except Exception as exc:
                log(f"speech recognition on the GPU failed ({exc}); using the CPU", "warn")
        if self.model is None:
            self.model = load(cfg.stt_quant or "int8", ["CPUExecutionProvider"])
            self.device = "cpu"
        self.model.recognize(np.zeros(MIC_RATE // 2, dtype=np.float32), sample_rate=MIC_RATE)  # warm up

    def transcribe(self, audio: np.ndarray) -> str:
        return str(self.model.recognize(audio, sample_rate=MIC_RATE)).strip()


class MoonshineSTT:
    """Moonshine streaming models by Moonshine AI (MIT): small and fast on a CPU.

    Medium (245M) is about as accurate as Whisper Large v3 at a sixth of the
    size. Key terms bias the decoder towards names like "Claude".
    """

    def __init__(self, cfg: Config, name: str = DEFAULT_STT_MODEL):
        from moonshine_voice import get_model_for_language, string_to_model_arch
        from moonshine_voice.transcriber import Transcriber

        self.cfg = cfg
        self.device = "cpu"
        language = (cfg.stt_language or "en").split("-")[0].lower()
        arch = self._arch_for(language, MOONSHINE_MODELS.get(name, MOONSHINE_MODELS[DEFAULT_STT_MODEL]))
        self.name = f"moonshine-{arch.split('-')[0]}"
        folder = os.path.join(os.path.expanduser("~/.cache/voice-agent/models"), "moonshine")
        shown = [0]

        def progress(fraction: float, *_):
            step = int(fraction * 4)  # a line at 25 %, 50 %, 75 % and done
            if step > shown[0]:
                shown[0] = step
                log(f"downloading {self.name}: {step * 25}%")

        path, model_arch = get_model_for_language(language, string_to_model_arch(arch),
                                                  cache_root=Path(folder), on_progress=progress)
        self.model = Transcriber(path, model_arch)
        terms = stt_keyterms(cfg)
        if terms:
            try:
                self.model.set_keyterms(terms)
            except Exception as exc:
                log(f"key terms unavailable ({exc})", "warn")
        self.transcribe(np.zeros(MIC_RATE // 2, dtype=np.float32))  # warm up

    @staticmethod
    def _arch_for(language: str, wanted: str) -> str:
        """The wanted streaming model, else the largest one this language has."""
        from moonshine_voice import ModelArch, model_arch_to_string
        from moonshine_voice.download import _stt_catalog

        catalog = _stt_catalog()
        if language not in catalog:
            raise RuntimeError(f"Moonshine has no model for language {language!r}; "
                               "choose parakeet-v3 (25 European languages)")
        have = {model_arch_to_string(ModelArch(m["model_arch"])) for m in catalog[language]["models"]}
        if wanted in have:
            return wanted
        for arch in MOONSHINE_MODELS.values():
            if arch in have:
                log(f"Moonshine {wanted.split('-')[0]} has no {language!r} model; using {arch.split('-')[0]}", "warn")
                return arch
        raise RuntimeError(f"Moonshine has no streaming model for language {language!r}; "
                           "choose parakeet-v3 (25 European languages)")

    def transcribe(self, audio: np.ndarray) -> str:
        transcript = self.model.transcribe_without_streaming(
            np.asarray(audio, dtype=np.float32).tolist(), sample_rate=MIC_RATE)
        return " ".join(line.text.strip() for line in transcript.lines if line.text and line.text.strip())


TURN_REPO = "pipecat-ai/smart-turn-v3"  # BSD 2-Clause
TURN_FILE = "smart-turn-v3.2-cpu.onnx"  # int8, about 9 MB
TURN_SECONDS = 8


def _mel_filters(n_mels: int = 80, n_fft: int = 400, rate: int = MIC_RATE) -> np.ndarray:
    """Slaney-style mel filter bank, as Whisper (and librosa) build it."""
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        lin = f / (200.0 / 3)
        log = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
        return np.where(f >= 1000.0, log, lin)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        lin = m * (200.0 / 3)
        log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
        return np.where(m >= 15.0, log, lin)

    fft_freqs = np.linspace(0, rate / 2, n_fft // 2 + 1)
    hz = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(rate / 2), n_mels + 2))
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (hz[2:] - hz[:-2]))[:, None]
    return weights.astype(np.float32)


_MEL = None


def whisper_features(audio: np.ndarray, seconds: int = TURN_SECONDS) -> np.ndarray:
    """Whisper log-mel features (80 x 100*seconds) of the last `seconds` of audio.

    Shorter audio is zero-padded at the front, as Smart Turn was trained.
    """
    global _MEL
    if _MEL is None:
        _MEL = _mel_filters()
    n = seconds * MIC_RATE
    audio = np.asarray(audio, dtype=np.float32)[-n:]
    if len(audio) < n:
        audio = np.concatenate([np.zeros(n - len(audio), dtype=np.float32), audio])
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
    n_fft, hop = 400, 160
    padded = np.pad(audio, n_fft // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:-1]
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = _MEL @ power.T.astype(np.float32)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


class SmartTurn:
    """Pipecat's Smart Turn v3: from the sound alone, has the speaker finished?

    A Whisper Tiny encoder with a small classifier head, int8 ONNX (9 MB):
    from about 10 ms per call on a recent CPU to 80 ms on an older laptop
    one. Returns the probability that the turn is complete.
    """

    def __init__(self, cfg: Config):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        folder = os.path.join(os.path.expanduser("~/.cache/voice-agent/models"), "smart-turn")
        path = os.path.join(folder, TURN_FILE)
        if not os.path.exists(path):
            path = hf_hub_download(TURN_REPO, TURN_FILE, local_dir=folder)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        # Two threads: it runs next to the transcription, which wants the rest.
        so.intra_op_num_threads = 2
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.predict(np.zeros(MIC_RATE, dtype=np.float32))  # warm up

    def predict(self, audio: np.ndarray) -> float:
        feats = whisper_features(audio)[None]
        out = self.session.run(None, {self.input: feats})[0]
        return float(np.asarray(out).reshape(-1)[0])


class PhononTTS:
    def __init__(self, cfg: Config):
        import ptts

        kwargs = {"lang": cfg.tts_lang or "en"}
        if cfg.tts_model:
            kwargs["model"] = cfg.tts_model
        if cfg.tts_quant:
            kwargs["quant"] = cfg.tts_quant
        if cfg.tts_voice:
            kwargs["voice"] = cfg.tts_voice
        self.tts = ptts.TTS(**kwargs)
        self.sample_rate = int(self.tts.sample_rate)

    def stream(self, text: str) -> Iterable[np.ndarray]:
        return self.tts.stream(text)


# --------------------------------------------------------------------------
# Speaking


class Speaker:
    """A queue of sentences, synthesized and played in order."""

    def __init__(self, tts, player: Player, on_state: Callable[[bool], None]):
        self.tts = tts
        self.player = player
        self.on_state = on_state
        self._q: "queue.Queue[tuple[int, str, Optional[threading.Event]]]" = queue.Queue()
        self._gen = 0
        self._lock = threading.Lock()
        self.speaking = False
        self.last_spoken = ""
        self.last_end = 0.0
        threading.Thread(target=self._run, daemon=True, name="speaker").start()

    def say(self, text: str, done: Optional[threading.Event] = None) -> None:
        text = text.strip()
        if not text:
            if done:
                done.set()
            return
        with self._lock:
            gen = self._gen
        self._q.put((gen, text, done))

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
        while True:
            try:
                _, _, done = self._q.get_nowait()
                if done:
                    done.set()
            except queue.Empty:
                break
        self.player.stop()

    def _current(self, gen: int) -> bool:
        with self._lock:
            return gen == self._gen

    def _set_speaking(self, value: bool) -> None:
        if self.speaking != value:
            self.speaking = value
            if not value:
                self.last_end = time.monotonic()
            self.on_state(value)

    def _run(self) -> None:
        while True:
            gen, text, done = self._q.get()
            if not self._current(gen):
                if done:
                    done.set()
                continue
            self._set_speaking(True)
            self.last_spoken = (self.last_spoken + " " + text)[-600:]
            try:
                stream = self.tts.stream(text)
                for chunk in stream:
                    if not self._current(gen):
                        break
                    if not self.player.write(pcm16(np.asarray(chunk, dtype=np.float32))):
                        break
                close = getattr(stream, "close", None)
                if close:
                    close()
            except Exception as exc:  # keep talking after one bad sentence
                emit("error", msg=f"tts failed: {exc}")
            if done:
                done.set()
            # Replies arrive sentence by sentence: give the next one a moment
            # to show up before closing the player.
            for _ in range(6):
                if not self._q.empty():
                    break
                time.sleep(0.05)
            if self._q.empty():
                if self._current(gen):
                    self.player.drain()
                self._set_speaking(False)


# --------------------------------------------------------------------------
# Listening


# Background noise floor for the loudness gate: its time constant, and a
# minimum so a silent room doesn't let any sound through.
NOISE_FLOOR_MS = 3000
NOISE_FLOOR_MIN = 0.002


class Listener:
    """Turns 30 ms mic frames into utterances.

    Hands-free capture uses the VAD; push-to-talk captures everything between
    down and up (plus a little audio from just before the key went down).

    A short pause mid-utterance calls on_pause with the audio so far, keyed by
    (utterance number, frames up to the pause). If the pause runs on into the
    end of the utterance, on_utterance gets the same key, so a transcript made
    during the pause can be used as is.
    """

    def __init__(self, cfg: Config, vad: Callable[[bytes], bool],
                 on_utterance: Callable[[bytes, str, Optional[tuple[int, int]]], None],
                 is_speaking: Callable[[], bool], on_barge_in: Callable[[], None], on_hearing: Callable[[bool], None],
                 on_pause: Callable[[bytes, tuple[int, int]], None] = lambda audio, key: None,
                 turn: Optional[Callable[[np.ndarray], float]] = None, check_text: bool = False):
        self.cfg = cfg
        self.vad = vad
        # Probability that the person has finished, from the audio so far.
        self.turn = turn
        # When the turn model says "finished", wait for the pause's early
        # transcript (see turn_text) before ending, in case it dangles.
        self.check_text = check_text
        self.turn_sync = False  # tests run the model inline
        # The pause being judged, and the answers for it as (key, verdict).
        self.awaiting: Optional[tuple[int, int]] = None
        self.turn_result: Optional[tuple[tuple[int, int], Optional[bool]]] = None
        self.text_result: Optional[tuple[tuple[int, int], bool]] = None
        self.on_utterance = on_utterance
        self.on_pause = on_pause
        self.utterances = 0
        self.is_speaking = is_speaking
        self.on_barge_in = on_barge_in
        self.on_hearing = on_hearing
        self.hands_free = True
        self.preroll: collections.deque[bytes] = collections.deque(maxlen=max(1, 450 // FRAME_MS))
        self.window: collections.deque[bool] = collections.deque(maxlen=10)
        self.buf: list[bytes] = []
        self.in_speech = False
        self.silence = 0
        self.voiced = 0
        self.loud_run = 0
        # Background loudness (RMS), and how many frames have shaped it.
        self.noise_floor = NOISE_FLOOR_MIN
        self.noise_frames = 0
        self.ptt = False
        self.ptt_buf: list[bytes] = []
        self.lock = threading.Lock()

    # push-to-talk
    def ptt_down(self) -> None:
        with self.lock:
            if self.ptt:
                return
            self.ptt = True
            self.ptt_buf = list(self.preroll)
            self._reset_vad()

    def ptt_up(self) -> None:
        with self.lock:
            if not self.ptt:
                return
            self.ptt = False
            audio = b"".join(self.ptt_buf)
            self.ptt_buf = []
        if len(audio) >= FRAME_BYTES * (self.cfg.min_utterance_ms // FRAME_MS):
            self.on_utterance(audio, "ptt", None)

    def _reset_vad(self) -> None:
        if self.in_speech:
            self.on_hearing(False)
        self.buf = []
        self.in_speech = False
        self.silence = 0
        self.window.clear()
        self.loud_run = 0

    def feed(self, frame: bytes) -> None:
        with self.lock:
            if self.ptt:
                self.ptt_buf.append(frame)
                self.preroll.append(frame)
                return
            hands_free = self.hands_free
        if not hands_free:
            self.preroll.append(frame)
            return

        speaking = self.is_speaking()
        voiced = raw_voiced = self.vad(frame)
        if speaking and self.cfg.barge_in and self.cfg.mode == "natural":
            # Our own voice leaks into the mic: only a loud, sustained voice
            # counts as the person cutting in.
            loud = voiced and rms(frame) >= self.cfg.barge_in_rms
            self.loud_run = self.loud_run + 1 if loud else 0
            if self.loud_run * FRAME_MS >= self.cfg.barge_in_ms:
                self.on_barge_in()
                self.loud_run = 0
                self._start(frame)
            else:
                self.preroll.append(frame)
            return

        level = rms(frame)
        if voiced and self.cfg.loudness_ratio > 0:
            voiced = level >= self.cfg.loudness_ratio * self.noise_floor
        # Our own voice leaking into the mic is not background noise.
        if not speaking and (not self.in_speech or not raw_voiced):
            self._track_noise(level)

        self.window.append(voiced)
        if not self.in_speech:
            self.preroll.append(frame)
            if sum(self.window) >= 6:
                self._start(None)
            return

        self.buf.append(frame)
        self.silence = 0 if voiced else self.silence + 1
        self.voiced += 1 if voiced else 0
        if voiced:
            self.awaiting = None
        too_long = len(self.buf) * FRAME_MS >= self.cfg.max_utterance_s * 1000
        speech_end = len(self.buf) - self.silence
        key = (self.utterances, speech_end)
        early = max(1, self.cfg.early_ms // FRAME_MS)
        if (self.silence == early and self.silence * FRAME_MS < self.cfg.silence_ms
                and self.voiced * FRAME_MS >= self.cfg.min_utterance_ms):
            audio = b"".join(self.buf)
            self.on_pause(audio, key)
            if self.turn is not None:
                self.awaiting = key
                self._ask_turn(audio, key)
        if self.silence * FRAME_MS >= self._silence_limit(key) or too_long:
            audio = b"".join(self.buf)
            voiced_ms = self.voiced * FRAME_MS
            self._reset_vad()
            if voiced_ms >= self.cfg.min_utterance_ms:
                self.on_utterance(audio, "voice", None if too_long else key)

    def _track_noise(self, level: float) -> None:
        """Fold one frame into the background noise floor.

        A running mean for the first few seconds, so the floor settles fast,
        then an exponential moving average over about NOISE_FLOOR_MS.
        """
        self.noise_frames += 1
        alpha = max(FRAME_MS / NOISE_FLOOR_MS, 1.0 / self.noise_frames)
        self.noise_floor = max(NOISE_FLOOR_MIN, self.noise_floor + alpha * (level - self.noise_floor))

    # ---- end-of-turn detection
    def _ask_turn(self, audio: bytes, key: tuple[int, int]) -> None:
        """Run the turn model on the audio so far, off the mic thread.

        It runs alongside the early transcription, which takes longer, so it
        adds nothing to the wait for the transcript.
        """
        def run():
            try:
                started = time.perf_counter()
                p = float(self.turn(np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0))
                done = p >= self.cfg.turn_threshold
                took = (time.perf_counter() - started) * 1000
                log(f"turn {'complete' if done else 'incomplete'} (p={p:.2f}, {took:.0f} ms)", "debug")
            except Exception as exc:
                log(f"turn detection failed ({exc}); using a fixed pause", "warn")
                done = None
            self.turn_result = (key, done)

        if self.turn_sync:
            run()
        else:
            threading.Thread(target=run, daemon=True, name="turn").start()

    def turn_text(self, key: tuple[int, int], text: str) -> None:
        """The early transcript of the pause `key`: does it end mid-sentence?"""
        self.text_result = (key, not ends_mid_sentence(text))

    def _silence_limit(self, key: tuple[int, int]) -> int:
        """How much silence ends the current pause."""
        if self.turn is None or self.awaiting != key:
            return self.cfg.silence_ms
        longest = max(self.cfg.silence_ms, self.cfg.turn_max_silence_ms)
        result, text = self.turn_result, self.text_result
        if result is None or result[0] != key:
            return longest  # the model has not answered yet: don't cut in
        if result[1] is None:
            return self.cfg.silence_ms  # the model failed: a fixed pause
        if not result[1]:
            return longest  # mid-thought: give them time
        if not self.check_text:
            return 0  # finished: end now
        if text is None or text[0] != key:
            # Finished, but the transcript is not in yet: end when it agrees,
            # or after the usual pause.
            return self.cfg.silence_ms
        return 0 if text[1] else longest

    def _start(self, frame: Optional[bytes]) -> None:
        self.awaiting = None
        self.in_speech = True
        self.utterances += 1
        self.silence = 0
        self.voiced = max(1, sum(self.window))
        self.buf = list(self.preroll)
        if frame is not None:
            self.buf.append(frame)
        self.preroll.clear()
        self.on_hearing(True)


# --------------------------------------------------------------------------
# The controller: modes, wake word, follow-ups, questions


@dataclass
class Ask:
    question: str
    answer: Optional[str] = None
    heard: str = ""
    retried: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    listening: bool = False


class Controller:
    def __init__(self, cfg: Config, stt, speaker_factory: Callable[["Controller"], Speaker],
                 chime: Callable[[str], None] = lambda name: None,
                 turn: Optional[Callable[[np.ndarray], float]] = None):
        self.cfg = cfg
        self.stt = stt
        self.chime = chime
        self.lock = threading.RLock()
        self.muted = False
        self.awake_until = 0.0
        self.ask: Optional[Ask] = None
        self.ask_lock = threading.Lock()
        self.state = "idle"
        self.hearing = False
        self.transcribing = 0
        # ("final", audio, source, during_speech, key) or ("early", audio, key)
        self.jobs: "queue.Queue[tuple]" = queue.Queue()
        self.early: tuple[Optional[tuple[int, int]], str] = (None, "")
        self.speaker = speaker_factory(self)
        self.listener = Listener(cfg, vad=lambda f: False, on_utterance=self._on_utterance,
                                 is_speaking=lambda: self.speaker.speaking,
                                 on_barge_in=self._on_barge_in, on_hearing=self._on_hearing,
                                 on_pause=self._on_pause, turn=turn,
                                 check_text=turn is not None and cfg.stt_language.startswith("en"))
        self._apply_mode()
        threading.Thread(target=self._transcribe_loop, daemon=True, name="stt").start()

    # ---- state
    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "mode": self.cfg.mode,
            "muted": self.muted,
            "awake": self._awake(),
            "asking": self.ask is not None,
            "ptt": self.listener.ptt,
        }

    def _publish(self) -> None:
        with self.lock:
            if self.muted and not self.listener.ptt:
                state = "muted"
            elif self.speaker.speaking:
                state = "speaking"
            elif self.listener.ptt or self.hearing:
                state = "hearing"
            elif self.transcribing:
                state = "transcribing"
            elif self.listener.hands_free:
                state = "listening"
            else:
                state = "idle"
            self.state = state
            snap = self.snapshot()
        emit("state", **snap)

    def _awake(self) -> bool:
        return time.monotonic() < self.awake_until

    def _apply_mode(self) -> None:
        with self.listener.lock:
            asking = self.ask is not None and self.ask.listening
            self.listener.hands_free = not self.muted and (self.cfg.mode != "ptt" or asking)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        with self.lock:
            self.cfg.mode = mode
            self.awake_until = 0.0
            self._apply_mode()
        self._publish()

    def set_muted(self, muted: bool) -> None:
        with self.lock:
            self.muted = muted
            self._apply_mode()
        self._publish()

    # ---- callbacks
    def on_speaking(self, speaking: bool) -> None:
        if not speaking:
            with self.lock:
                if self.cfg.mode == "wake":
                    # Let the person answer without saying the wake word again.
                    self.awake_until = time.monotonic() + self.cfg.follow_up_s
                if self.ask is not None and not self.ask.listening:
                    self.ask.listening = True
                    self._apply_mode()
        self._publish()

    def _on_hearing(self, hearing: bool) -> None:
        self.hearing = hearing
        self._publish()

    def _on_barge_in(self) -> None:
        self.speaker.stop()
        emit("barge_in")

    def _on_utterance(self, audio: bytes, source: str, key: Optional[tuple[int, int]] = None) -> None:
        during_speech = self.speaker.speaking or time.monotonic() - self.speaker.last_end < 0.6
        self.jobs.put(("final", audio, source, during_speech, key))

    def _on_pause(self, audio: bytes, key: tuple[int, int]) -> None:
        self.jobs.put(("early", audio, key))

    # ---- push-to-talk
    def ptt(self, action: str) -> None:
        if action == "toggle":
            action = "up" if self.listener.ptt else "down"
        if action == "down":
            if self.speaker.speaking:
                self._on_barge_in()
            self.listener.ptt_down()
            self.chime("ptt_down")
        elif action == "up":
            self.chime("ptt_up")
            self.listener.ptt_up()
        else:
            raise ValueError(f"unknown push-to-talk action {action!r}")
        self._publish()

    # ---- transcription
    def _transcribe(self, audio: bytes) -> str:
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        return self.stt.transcribe(pcm)

    def _transcribe_loop(self) -> None:
        while True:
            job = self.jobs.get()
            if job[0] == "early":
                _, audio, key = job
                if not self.jobs.empty():
                    continue  # stale already: the final one is waiting
                try:
                    self.early = (key, self._transcribe(audio))
                except Exception:
                    self.early = (None, "")
                    continue
                self.listener.turn_text(key, self.early[1])
                continue
            _, audio, source, during_speech, key = job
            self.transcribing += 1
            self._publish()
            try:
                early_key, early_text = self.early
                if key is not None and key == early_key:
                    text = early_text
                else:
                    text = self._transcribe(audio)
            except Exception as exc:
                emit("error", msg=f"transcription failed: {exc}")
                text = ""
            finally:
                self.transcribing -= 1
                self.early = (None, "")
            try:
                self.handle_text(text, source, during_speech)
            except Exception as exc:
                emit("error", msg=f"handling speech failed: {exc}")
            self._publish()

    def handle_text(self, text: str, source: str, during_speech: bool = False) -> None:
        """Decide what one transcribed utterance is for."""
        text = text.strip()
        if is_hallucination(text):
            return
        if source != "ptt" and during_speech and is_echo(text, self.speaker.last_spoken):
            return

        ask = self.ask
        if ask is not None:
            rest = match_wake(text, self._wake_phrases(), self.cfg.wake_threshold)
            answer = classify_yes_no(rest if rest is not None else text)
            if answer == "unclear" and not ask.retried:
                ask.retried = True
                ask.listening = False
                self._apply_mode()
                self.speaker.say("Sorry, was that a yes or a no?")
                return
            ask.heard = text
            ask.answer = answer
            ask.done.set()
            return

        if source == "ptt" or self.cfg.mode == "natural":
            self._deliver(text, source)
            return

        if self.cfg.mode == "wake":
            rest = match_wake(text, self._wake_phrases(), self.cfg.wake_threshold)
            if rest is not None:
                if during_speech:
                    self.speaker.stop()
                    emit("barge_in")
                if rest:
                    self._deliver(rest, "wake")
                else:
                    with self.lock:
                        self.awake_until = time.monotonic() + self.cfg.follow_up_s
                    self.chime("wake")
                    emit("wake")
                return
            if self._awake() and not during_speech:
                self._deliver(text, "follow_up")
            return
        # ptt mode: hands-free speech is ignored

    def _deliver(self, text: str, source: str) -> None:
        with self.lock:
            if self.cfg.mode == "wake":
                self.awake_until = 0.0
        emit("transcript", text=text, source=source)

    def _wake_phrases(self) -> list[str]:
        return [self.cfg.wake_word, *self.cfg.wake_aliases]

    # ---- spoken yes/no questions
    def ask_question(self, question: str, timeout: float) -> dict:
        with self.ask_lock:  # one question at a time
            ask = Ask(question=question)
            with self.lock:
                self.ask = ask
            self.chime("ask")
            self.speaker.say(question)
            # If nothing is speaking (TTS failed or empty), listen right away.
            time.sleep(0.05)
            if not self.speaker.speaking and self.speaker._q.empty():
                ask.listening = True
            self._apply_mode()
            self._publish()
            answered = ask.done.wait(timeout)
            with self.lock:
                self.ask = None
                self._apply_mode()
            self._publish()
            if not answered:
                return {"answer": "timeout", "heard": ""}
            if ask.answer == "yes":
                self.speaker.say("Okay.")
            elif ask.answer == "no":
                self.speaker.say("Alright, I won't.")
            return {"answer": ask.answer, "heard": ask.heard}


# --------------------------------------------------------------------------
# Control socket


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(ctl: Controller, shutdown: Callable[[], None]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def address_string(self):  # Unix sockets have no peer address
            return "local"

        def log_message(self, *args):
            pass

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                self._send(200, ctl.snapshot())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            body = self._body()
            path = self.path.rstrip("/")
            try:
                if path == "/speak":
                    if body.get("interrupt"):
                        ctl.speaker.stop()
                    ctl.speaker.say(str(body.get("text", "")))
                elif path == "/stop":
                    ctl.speaker.stop()
                elif path == "/mode":
                    ctl.set_mode(str(body.get("mode", "")))
                elif path == "/mute":
                    ctl.set_muted(bool(body.get("muted", True)))
                elif path.startswith("/ptt/"):
                    ctl.ptt(path.split("/")[-1])
                elif path == "/ask":
                    timeout = float(body.get("timeout", 45))
                    self._send(200, ctl.ask_question(str(body.get("question", "")), min(max(timeout, 3), 300)))
                    return
                elif path == "/shutdown":
                    self._send(200, {"ok": True})
                    threading.Thread(target=shutdown, daemon=True).start()
                    return
                else:
                    self._send(404, {"error": "not found"})
                    return
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(200, {"ok": True, **ctl.snapshot()})

    return Handler


def take_over_socket(path: str) -> None:
    """If another helper owns the socket, ask it to quit; then free the path."""
    if not os.path.exists(path):
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(path)
            s.sendall(b"POST /shutdown HTTP/1.1\r\nHost: voice\r\nContent-Length: 0\r\n\r\n")
            s.recv(256)
        log("another voice session was using the microphone; it has been stopped")
        for _ in range(30):
            if not os.path.exists(path):
                break
            time.sleep(0.1)
    except OSError:
        pass
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------
# main


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--socket", required=True)
    ap.add_argument("--config", default="{}", help="JSON object of Config fields")
    args = ap.parse_args(argv)
    cfg = Config.from_json(args.config)

    stop_event = threading.Event()

    def shutdown(*_):
        stop_event.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        emit("state", state="loading", mode=cfg.mode, muted=False, awake=False, asking=False, ptt=False)
        log(f"loading speech recognition ({resolve_stt_model(cfg.stt_model)}); the first run downloads it")
        stt = make_stt(cfg)
        log(f"speech recognition ready on the {stt.device.upper()}")
        turn = None
        if cfg.turn_detection:
            try:
                turn = SmartTurn(cfg).predict
                log("turn detection ready (Smart Turn v3)")
            except Exception as exc:
                log(f"turn detection unavailable ({exc}); an utterance ends after {cfg.silence_ms} ms of silence", "warn")
        log("loading Phonon voice; the first run downloads it")
        tts = PhononTTS(cfg)
        import webrtcvad

        vad_engine = webrtcvad.Vad(min(3, max(0, int(cfg.vad_aggressiveness))))
        player_argv = playback_command(cfg, tts.sample_rate)
        chime_argv = playback_command(cfg, 24_000)
        mic_argv = capture_command(cfg)
    except Exception as exc:
        emit("error", msg=f"startup failed: {exc}", fatal=True)
        return 1

    chimes = {
        "wake": tone([660, 880], 24_000),
        "ptt_down": tone([880], 24_000, ms=60),
        "ptt_up": tone([520], 24_000, ms=60),
        "ask": tone([740, 740], 24_000, ms=70),
    }

    def chime(name: str) -> None:
        if cfg.chimes and name in chimes:
            play_once(chime_argv, chimes[name])

    ctl = Controller(cfg, stt, lambda c: Speaker(tts, Player(player_argv), c.on_speaking), chime=chime, turn=turn)
    ctl.listener.vad = lambda frame: vad_engine.is_speech(frame, MIC_RATE)

    take_over_socket(args.socket)
    os.makedirs(os.path.dirname(args.socket) or ".", exist_ok=True)
    old_umask = os.umask(0o177)
    try:
        server = _UnixHTTPServer(args.socket, make_handler(ctl, shutdown))
    finally:
        os.umask(old_umask)
    threading.Thread(target=server.serve_forever, daemon=True, name="control").start()

    mic = subprocess.Popen(mic_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read_mic():
        assert mic.stdout is not None
        while not stop_event.is_set():
            frame = mic.stdout.read(FRAME_BYTES)
            if not frame or len(frame) < FRAME_BYTES:
                err = mic.stderr.read().decode(errors="replace").strip() if mic.stderr else ""
                emit("error", msg=f"microphone stopped ({' '.join(mic_argv[:1])}): {err[:300]}", fatal=True)
                stop_event.set()
                return
            try:
                ctl.listener.feed(frame)
            except Exception as exc:
                emit("error", msg=f"listening failed: {exc}")

    threading.Thread(target=read_mic, daemon=True, name="mic").start()

    emit("ready", socket=args.socket, mode=cfg.mode, sample_rate=tts.sample_rate,
         recorder=mic_argv[0], player=player_argv[0])
    ctl._publish()

    # Exit with the mod: its pipe closing makes emit() exit, and a changed
    # parent means it died without closing anything.
    parent = os.getppid()
    beat = 0
    while not stop_event.wait(0.5):
        if os.getppid() != parent:
            break
        beat += 1
        if beat % 10 == 0:
            emit("ping")

    ctl.speaker.stop()
    mic.kill()
    server.shutdown()
    server.server_close()
    try:
        os.unlink(args.socket)
    except FileNotFoundError:
        pass
    emit("state", state="stopped", mode=cfg.mode, muted=False, awake=False, asking=False, ptt=False)
    return 0


def relaunch_on_gpu() -> None:
    """Re-run this script under uv with the CUDA build of onnxruntime layered on.

    Only for an NVIDIA machine without it: the first time, uv downloads the
    CUDA runtime (about 2 GB). If anything goes wrong, carry on on the CPU.
    """
    uv = shutil.which("uv")
    if not uv or not shutil.which("nvidia-smi") or os.environ.get("VOICE_AGENT_GPU") or cuda_available():
        return
    cmd = [uv, "run", "--quiet", "--with", GPU_RUNTIME, "--script", os.path.abspath(__file__)]
    env = {**os.environ, "VOICE_AGENT_GPU": "1"}
    log("setting up GPU speech recognition; the first run downloads the CUDA runtime (about 2 GB)")
    try:
        probe = subprocess.run([*cmd, "--check-gpu"], env=env, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"no GPU runtime ({exc}); using the CPU", "warn")
        return
    if probe.returncode != 0 or "cuda" not in probe.stdout:
        log(f"no GPU runtime ({(probe.stderr or probe.stdout).strip()[-200:]}); using the CPU", "warn")
        return
    sys.stdout.flush()
    os.execvpe(uv, [*cmd, *sys.argv[1:]], env)


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-gpu"]:
        print("cuda" if cuda_available() else "cpu")
        sys.exit(0)
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--config", default="{}")
    _cfg = Config.from_json(_ap.parse_known_args()[0].config)
    # Only Parakeet runs on the GPU: Moonshine never needs the CUDA download.
    if _cfg.stt_device != "cpu" and is_parakeet(resolve_stt_model(_cfg.stt_model)):
        emit("state", state="loading", mode=_cfg.mode, muted=False, awake=False, asking=False, ptt=False)
        relaunch_on_gpu()
    sys.exit(main())
