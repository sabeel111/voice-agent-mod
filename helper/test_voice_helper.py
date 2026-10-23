"""Tests for voice_helper.py with fake speech engines (no models, no audio devices).

Run: uv run --with numpy --with webrtcvad-wheels --with pytest pytest helper -q
"""

from __future__ import annotations

import http.client
import json
import os
import queue
import socket
import sys
import tempfile
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import voice_helper as vh  # noqa: E402


# --------------------------------------------------------------------------
# fakes


class Events:
    def __init__(self):
        self.items: list[dict] = []
        self.cv = threading.Condition()

    def __call__(self, kind, **payload):
        with self.cv:
            self.items.append({"type": kind, **payload})
            self.cv.notify_all()

    def of(self, kind):
        return [e for e in self.items if e["type"] == kind]

    def wait_for(self, kind, count=1, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self.cv:
            while len(self.of(kind)) < count:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise AssertionError(f"no {kind} event; got {self.items}")
                self.cv.wait(left)
        return self.of(kind)


@pytest.fixture
def events(monkeypatch):
    ev = Events()
    monkeypatch.setattr(vh, "emit", ev)
    return ev


class FakeTTS:
    sample_rate = 24_000

    def __init__(self, seconds=0.05):
        self.seconds = seconds
        self.said: list[str] = []

    def stream(self, text):
        self.said.append(text)
        n = int(self.sample_rate * self.seconds)
        for _ in range(2):
            yield np.zeros(n // 2, dtype=np.float32)


class FakePlayer(vh.Player):
    def __init__(self):
        self.played = 0
        self.stopped = 0

    def write(self, data):
        self.played += len(data)
        time.sleep(0.005)
        return True

    def drain(self):
        pass

    def stop(self):
        self.stopped += 1


class ScriptedSTT:
    """Returns queued texts in order, ignoring the audio."""

    def __init__(self):
        self.texts: list[str] = []
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.texts.pop(0) if self.texts else ""


def make_controller(mode="wake", **overrides):
    cfg = vh.Config(mode=mode, **overrides)
    tts = FakeTTS()
    player = FakePlayer()
    stt = ScriptedSTT()
    chimes: list[str] = []
    ctl = vh.Controller(cfg, stt, lambda c: vh.Speaker(tts, player, c.on_speaking), chime=chimes.append)
    return ctl, stt, tts, player, chimes


def wait_until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


# --------------------------------------------------------------------------
# pure helpers


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hey Claude, fix the login bug.", "fix the login bug"),
        ("hey cloud what's the status", "what's the status"),
        ("Um, hey Claude. Can you check the tests?", "Can you check the tests"),
        ("Hey, Clyde!", ""),
        ("Hey Claude", ""),
        ("Okay Claude, stop.", "stop"),
        ("I was telling Claude about it", None),
        ("what's the weather", None),
        ("", None),
    ],
)
def test_match_wake(text, expected):
    phrases = ["hey claude", "hey cloud", "hey clyde", "okay claude"]
    assert vh.match_wake(text, phrases, 0.75) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes", "yes"),
        ("yeah go ahead", "yes"),
        ("Sure, do it.", "yes"),
        ("No.", "no"),
        ("no, don't do that", "no"),
        ("Hold on a second", "no"),
        ("Not now", "no"),
        ("what does it want to do?", "unclear"),
        ("", "unclear"),
    ],
)
def test_classify_yes_no(text, expected):
    assert vh.classify_yes_no(text) == expected


def test_echo_detection():
    spoken = "I'll hand that off to an agent now. It should take a minute."
    assert vh.is_echo("hand that off to an agent now", spoken)
    assert not vh.is_echo("wait, use the other database", spoken)


def test_garbled_echo_still_matches_what_was_said():
    spoken = "Here is a longer answer to test the fix. I should get through this whole reply."
    assert vh.is_echo("here is a longest answer to test the fixed", spoken)
    assert not vh.is_echo("can you open the settings file instead", spoken)


@pytest.mark.parametrize(
    "text,short",
    [("important", True), ("I love that.", False), ("Oh, nice.", True), ("Stop.", False),
     ("Hold on!", False), ("Hey Claude", False), ("Wait, use the other database.", False)],
)
def test_too_short_to_interrupt(text, short):
    assert vh.too_short_to_interrupt(text) is short


def test_hallucinations_are_dropped():
    assert vh.is_hallucination("Thank you.")
    assert vh.is_hallucination("...")
    assert not vh.is_hallucination("run the tests")


def test_stt_model_names():
    assert vh.resolve_stt_model("") == vh.DEFAULT_STT_MODEL == "moonshine-medium"
    assert vh.resolve_stt_model("moonshine-small") == "moonshine-small"
    assert vh.resolve_stt_model("parakeet-v2") == vh.PARAKEET_DEFAULT
    assert vh.resolve_stt_model("parakeet-v3") == "nemo-parakeet-tdt-0.6b-v3"
    assert vh.resolve_stt_model("base.en") == vh.DEFAULT_STT_MODEL  # an old Whisper setting
    assert vh.is_parakeet(vh.resolve_stt_model("parakeet-v3"))
    assert not vh.is_parakeet(vh.resolve_stt_model("moonshine-tiny"))


def test_make_stt_picks_the_engine(monkeypatch):
    monkeypatch.setattr(vh, "ParakeetSTT", lambda cfg: ("parakeet", cfg.stt_model))
    monkeypatch.setattr(vh, "MoonshineSTT", lambda cfg, name: ("moonshine", name))
    assert vh.make_stt(vh.Config()) == ("moonshine", "moonshine-medium")
    assert vh.make_stt(vh.Config(stt_model="moonshine-small")) == ("moonshine", "moonshine-small")
    assert vh.make_stt(vh.Config(stt_model="parakeet-v3")) == ("parakeet", "parakeet-v3")


def test_keyterms_lead_with_the_wake_word():
    cfg = vh.Config(wake_word="hey claude", stt_keyterms=["Omarchy", "  Hyprland   config ", "claude", ""])
    assert vh.stt_keyterms(cfg) == ["Claude", "Omarchy", "Hyprland config"]
    assert vh.stt_keyterms(vh.Config(wake_word="okay jarvis")) == ["Jarvis"]


class FakeMoonshine:
    """Stands in for the moonshine_voice package: a catalog, a download and a transcriber."""

    CATALOG = {
        "en": {"english_name": "English", "models": [{"model_arch": 5}, {"model_arch": 4}, {"model_arch": 2}]},
        "de": {"english_name": "German", "models": [{"model_arch": 4}, {"model_arch": 2}]},
        "ko": {"english_name": "Korean", "models": [{"model_arch": 0}]},
    }

    def __init__(self, monkeypatch, lines=("Hey Claude,", " check the build. ")):
        import enum
        import types

        class ModelArch(enum.IntEnum):
            TINY = 0
            BASE = 1
            TINY_STREAMING = 2
            BASE_STREAMING = 3
            SMALL_STREAMING = 4
            MEDIUM_STREAMING = 5

        names = {0: "tiny", 1: "base", 2: "tiny-streaming", 3: "base-streaming", 4: "small-streaming", 5: "medium-streaming"}
        fake = self
        self.downloads: list[tuple] = []
        self.keyterms: list[list[str]] = []
        self.audio: list[int] = []

        class Line:
            def __init__(self, text):
                self.text = text

        class Transcript:
            def __init__(self, texts):
                self.lines = [Line(t) for t in texts]

        class Transcriber:
            def __init__(self, path, arch):
                fake.loaded = (path, arch)

            def set_keyterms(self, terms):
                fake.keyterms.append(list(terms))

            def transcribe_without_streaming(self, audio, sample_rate=16000):
                assert sample_rate == vh.MIC_RATE and isinstance(audio, list)
                fake.audio.append(len(audio))
                return Transcript(lines if len(fake.audio) > 1 else [])

        def get_model_for_language(language, arch, *, cache_root=None, on_progress=None):
            fake.downloads.append((language, names[int(arch)], str(cache_root)))
            if on_progress:
                for f in (0.1, 0.3, 0.6, 1.0):
                    on_progress(f, "model")
            return f"/models/{language}/{names[int(arch)]}", arch

        root = types.ModuleType("moonshine_voice")
        root.ModelArch = ModelArch
        root.model_arch_to_string = lambda a: names[int(a)]
        root.string_to_model_arch = lambda s: ModelArch({v: k for k, v in names.items()}[s])
        root.get_model_for_language = get_model_for_language
        download = types.ModuleType("moonshine_voice.download")
        download._stt_catalog = lambda: FakeMoonshine.CATALOG
        transcriber = types.ModuleType("moonshine_voice.transcriber")
        transcriber.Transcriber = Transcriber
        for name, mod in (("moonshine_voice", root), ("moonshine_voice.download", download),
                          ("moonshine_voice.transcriber", transcriber)):
            monkeypatch.setitem(sys.modules, name, mod)


def test_moonshine_loads_transcribes_and_biases(events, monkeypatch):
    fake = FakeMoonshine(monkeypatch)
    stt = vh.MoonshineSTT(vh.Config(stt_keyterms=["Omarchy"]), "moonshine-medium")
    assert stt.name == "moonshine-medium" and stt.device == "cpu"
    language, arch, cache = fake.downloads[0]
    assert (language, arch) == ("en", "medium-streaming")
    assert cache.endswith(".cache/voice-agent/models/moonshine")
    assert fake.keyterms == [["Claude", "Omarchy"]]
    assert fake.audio == [vh.MIC_RATE // 2]  # warmed up
    assert [e["msg"] for e in events.of("log")] == [f"downloading moonshine-medium: {p}%" for p in (25, 50, 100)]
    assert stt.transcribe(np.zeros(8000, dtype=np.float32)) == "Hey Claude, check the build."


def test_moonshine_falls_back_to_a_model_the_language_has(events, monkeypatch):
    fake = FakeMoonshine(monkeypatch)
    stt = vh.MoonshineSTT(vh.Config(stt_language="de"), "moonshine-medium")
    assert fake.downloads[0][:2] == ("de", "small-streaming") and stt.name == "moonshine-small"
    assert any("no 'de' model; using small" in e["msg"] for e in events.of("log"))
    with pytest.raises(RuntimeError, match="parakeet-v3"):
        vh.MoonshineSTT(vh.Config(stt_language="ko"), "moonshine-small")
    with pytest.raises(RuntimeError, match="parakeet-v3"):
        vh.MoonshineSTT(vh.Config(stt_language="xx"), "moonshine-small")


def test_early_transcript_is_reused_when_the_pause_ends_the_utterance(events):
    ctl, stt, *_ = make_controller("natural")
    stt.texts = ["add a test"]
    ctl._on_pause(b"\0\0" * 1600, (1, 40))
    wait_until(lambda: ctl.early[0] == (1, 40))
    ctl._on_utterance(b"\0\0" * 3200, "voice", (1, 40))
    assert events.wait_for("transcript")[0]["text"] == "add a test"
    assert stt.calls == 1


def test_early_transcript_is_dropped_when_speech_went_on(events):
    ctl, stt, *_ = make_controller("natural")
    stt.texts = ["add a", "add a test please"]
    ctl._on_pause(b"\0\0" * 1600, (1, 40))
    wait_until(lambda: stt.calls == 1)
    ctl._on_utterance(b"\0\0" * 3200, "voice", (1, 70))
    assert events.wait_for("transcript")[0]["text"] == "add a test please"


def test_config_from_json_ignores_unknown_and_bad_mode():
    cfg = vh.Config.from_json(json.dumps({"mode": "bogus", "wake_word": "hey jarvis", "nope": 1, "tts_voice": ""}))
    assert cfg.mode == "wake"
    assert cfg.wake_word == "hey jarvis"
    assert cfg.tts_voice == ""


def test_tone_is_pcm16():
    data = vh.tone([440], 24_000, ms=50)
    assert len(data) == 2 * int(24_000 * 0.05)


# --------------------------------------------------------------------------
# listener (VAD segmentation and push-to-talk)


def voiced_frame():
    return (np.ones(vh.FRAME_SAMPLES, dtype=np.int16) * 3000).tobytes()


def silent_frame():
    return np.zeros(vh.FRAME_SAMPLES, dtype=np.int16).tobytes()


def make_listener(mode="natural", speaking=lambda: False, pauses=None, checks=None):
    got: list[tuple[int, str]] = []
    barge: list[int] = []
    cfg = vh.Config(mode=mode, silence_ms=300, early_ms=90, min_utterance_ms=200, loudness_ratio=0)
    lis = vh.Listener(
        cfg,
        vad=lambda f: vh.rms(f) > 0.01,
        on_utterance=lambda audio, src, key: got.append((len(audio), src, key)),
        is_speaking=speaking,
        on_barge_in=lambda: barge.append(1),
        on_hearing=lambda h: None,
        on_pause=(lambda audio, key: pauses.append(key)) if pauses is not None else (lambda audio, key: None),
        on_verify=(lambda audio, n: checks.append((len(audio), n))) if checks is not None else (lambda audio, n: None),
    )
    return lis, got, barge


def frame_at(amplitude):
    return (np.ones(vh.FRAME_SAMPLES, dtype=np.int16) * amplitude).tobytes()


def make_gated_listener(ratio):
    """A listener whose VAD calls everything speech, so only loudness decides."""
    got: list = []
    cfg = vh.Config(mode="natural", silence_ms=300, early_ms=90, min_utterance_ms=200,
                    turn_detection=False, loudness_ratio=ratio)
    lis = vh.Listener(
        cfg,
        vad=lambda f: True,
        on_utterance=lambda audio, src, key: got.append((len(audio), src, key)),
        is_speaking=lambda: False,
        on_barge_in=lambda: None,
        on_hearing=lambda h: None,
    )
    return lis, got


def test_listener_ignores_voices_not_louder_than_the_background():
    lis, got = make_gated_listener(2.0)
    for _ in range(100):  # 3 s of background chatter
        lis.feed(frame_at(500))
    for _ in range(30):  # a bit louder, but under twice the background
        lis.feed(frame_at(800))
    for _ in range(15):
        lis.feed(frame_at(500))
    assert got == []
    assert not lis.in_speech


def test_listener_hears_speech_well_above_the_background():
    lis, got = make_gated_listener(2.0)
    for _ in range(100):
        lis.feed(frame_at(500))
    for _ in range(30):
        lis.feed(frame_at(3000))
    for _ in range(15):
        lis.feed(frame_at(500))  # back to background: the utterance ends
    assert len(got) == 1 and got[0][1] == "voice"


def test_listener_loudness_ratio_zero_turns_the_gate_off():
    lis, got = make_gated_listener(0)
    for _ in range(100):
        lis.feed(frame_at(500))
    assert lis.in_speech


def test_listener_pause_key_matches_the_utterance_that_follows():
    pauses: list = []
    lis, got, _ = make_listener(pauses=pauses)
    for _ in range(30):
        lis.feed(voiced_frame())
    for _ in range(15):
        lis.feed(silent_frame())
    assert len(pauses) == 1 and len(got) == 1
    assert got[0][2] == pauses[0]


def test_listener_resumed_speech_gets_a_new_pause_key():
    pauses: list = []
    lis, got, _ = make_listener(pauses=pauses)
    for _ in range(30):
        lis.feed(voiced_frame())
    for _ in range(5):  # a short pause, then more words
        lis.feed(silent_frame())
    for _ in range(20):
        lis.feed(voiced_frame())
    for _ in range(15):
        lis.feed(silent_frame())
    assert len(pauses) == 2 and len(got) == 1
    assert got[0][2] == pauses[1] != pauses[0]


def test_listener_segments_an_utterance():
    lis, got, _ = make_listener()
    for _ in range(20):
        lis.feed(silent_frame())
    for _ in range(30):
        lis.feed(voiced_frame())
    for _ in range(15):
        lis.feed(silent_frame())
    assert len(got) == 1
    assert got[0][1] == "voice"
    # pre-roll + speech + trailing silence
    assert got[0][0] >= 30 * vh.FRAME_BYTES


def test_listener_ignores_short_blips():
    lis, got, _ = make_listener()
    for _ in range(6):  # 180 ms of voice, under the 200 ms minimum
        lis.feed(voiced_frame())
    for _ in range(20):
        lis.feed(silent_frame())
    assert got == []


def test_push_to_talk_captures_between_down_and_up():
    lis, got, _ = make_listener(mode="ptt")
    lis.hands_free = False
    for _ in range(10):
        lis.feed(silent_frame())
    lis.ptt_down()
    for _ in range(20):
        lis.feed(voiced_frame())
    lis.ptt_up()
    assert len(got) == 1 and got[0][1] == "ptt"
    # includes the pre-roll from before the key went down
    assert got[0][0] == 30 * vh.FRAME_BYTES


def test_barge_in_needs_sustained_loud_voice_in_natural_mode():
    checks: list = []
    lis, got, barge = make_listener(speaking=lambda: True, checks=checks)
    lis.cfg.barge_in_rms = 0.05
    quiet = (np.ones(vh.FRAME_SAMPLES, dtype=np.int16) * 800).tobytes()  # echo-level
    for _ in range(40):
        lis.feed(quiet)
    assert barge == [] and checks == []
    loud = (np.ones(vh.FRAME_SAMPLES, dtype=np.int16) * 6000).tobytes()
    for _ in range(30):
        lis.feed(loud)
    # loud enough to check, but not taken until the words are
    assert barge == [] and len(checks) == 1
    assert checks[0][0] >= vh.BARGE_IN_VERIFY_MS // vh.FRAME_MS * vh.FRAME_BYTES
    lis.verify_result(checks[0][1], True)
    lis.feed(loud)
    assert barge == [1]


def test_barge_in_that_checks_out_as_echo_is_dropped():
    checks: list = []
    lis, got, barge = make_listener(speaking=lambda: True, checks=checks)
    loud = (np.ones(vh.FRAME_SAMPLES, dtype=np.int16) * 6000).tobytes()
    for _ in range(30):
        lis.feed(loud)
    lis.verify_result(checks[0][1], False)
    lis.feed(loud)
    assert barge == [] and not lis.in_speech
    for _ in range(20):
        lis.feed(silent_frame())
    assert got == []
    # and the next loud run gets a check of its own
    for _ in range(30):
        lis.feed(loud)
    assert len(checks) == 2 and checks[1][1] != checks[0][1]


# --------------------------------------------------------------------------
# controller: modes, wake word, follow-ups


def test_wake_mode_delivers_command_after_wake_word(events):
    ctl, *_ = make_controller("wake")
    ctl.handle_text("Hey Claude, refactor the auth module", "voice")
    assert events.of("transcript")[-1]["text"] == "refactor the auth module"
    assert events.of("transcript")[-1]["source"] == "wake"


def test_wake_mode_ignores_speech_without_wake_word(events):
    ctl, *_ = make_controller("wake")
    ctl.handle_text("I think the TV is too loud", "voice")
    assert events.of("transcript") == []


def test_wake_word_alone_opens_a_follow_up_window(events):
    ctl, _, _, _, chimes = make_controller("wake")
    ctl.handle_text("Hey Claude.", "voice")
    assert events.of("wake") and "wake" in chimes
    ctl.handle_text("What are the agents doing?", "voice")
    assert events.of("transcript")[-1]["text"] == "What are the agents doing?"
    # the window closes once used
    ctl.handle_text("and something else", "voice")
    assert len(events.of("transcript")) == 1


def test_follow_up_window_opens_after_claude_speaks(events):
    ctl, _, tts, _, _ = make_controller("wake")
    ctl.speaker.say("Want me to add tests too?")
    wait_until(lambda: tts.said and not ctl.speaker.speaking and ctl._awake())
    ctl.handle_text("Yes, add tests.", "voice")
    assert events.of("transcript")[-1]["text"] == "Yes, add tests."


def test_natural_mode_delivers_everything(events):
    ctl, *_ = make_controller("natural")
    ctl.handle_text("let's talk about the release", "voice")
    assert events.of("transcript")[-1]["source"] == "voice"


def test_ptt_mode_ignores_hands_free_but_takes_ptt(events):
    ctl, *_ = make_controller("ptt")
    assert ctl.listener.hands_free is False
    ctl.handle_text("hands free speech", "voice")
    assert events.of("transcript") == []
    ctl.handle_text("pushed to talk", "ptt")
    assert events.of("transcript")[-1]["text"] == "pushed to talk"


def test_echo_of_our_own_voice_is_dropped(events):
    ctl, *_ = make_controller("natural")
    ctl.speaker.last_spoken = "The refactor is done and all tests pass."
    ctl.handle_text("the refactor is done and all tests pass", "voice", during_speech=True)
    assert events.of("transcript") == []


def test_echo_phantoms_are_dropped_only_around_our_speech(events):
    assert vh.is_echo_phantom("Yeah, thank you.")
    assert not vh.is_echo_phantom("yeah")
    ctl, *_ = make_controller("natural")
    ctl.handle_text("Yeah, thank you.", "voice", during_speech=True)
    assert events.of("transcript") == []
    ctl.handle_text("Yeah, thank you.", "voice")
    assert events.of("transcript")[-1]["text"] == "Yeah, thank you."


def test_echo_tail_counts_as_during_speech():
    ctl, *_ = make_controller("natural")
    ctl.jobs = queue.Queue()  # out of the transcription thread's reach
    ctl.speaker.last_end = time.monotonic() - 1.0  # Bluetooth lag: still echo
    ctl._on_utterance(b"", "voice")
    assert ctl.jobs.get(timeout=1)[3] is True
    ctl.speaker.last_end = time.monotonic() - 2.0
    ctl._on_utterance(b"", "voice")
    assert ctl.jobs.get(timeout=1)[3] is False


def test_a_word_or_two_over_our_own_voice_is_dropped(events):
    ctl, *_ = make_controller("natural")
    ctl.speaker.last_spoken = "Here is a longer answer to test the fix."
    ctl.handle_text("important", "voice", during_speech=True)
    ctl.handle_text("Oh, nice.", "voice", during_speech=True)
    assert events.of("transcript") == []
    ctl.handle_text("Stop.", "voice", during_speech=True)  # a request to stop is short
    ctl.handle_text("important", "voice")  # nothing playing: a one-word reply is fine
    ctl.handle_text("ok", "ptt", during_speech=True)
    assert [e["text"] for e in events.of("transcript")] == ["Stop.", "important", "ok"]


def test_short_answers_to_a_question_are_not_held_to_the_overlap_rule(events):
    ctl, *_ = make_controller("natural")
    ctl.ask = vh.Ask("Run the tests?", listening=True)
    ctl.handle_text("Yes.", "voice", during_speech=True)
    assert ctl.ask.answer == "yes"


def test_an_utterance_that_began_in_the_echo_tail_counts_as_during_speech():
    ctl, *_ = make_controller("natural")
    ctl.jobs = queue.Queue()
    ctl.speaker.last_end = time.monotonic() - 2.5  # ended long ago...
    two_seconds = bytes(2 * vh.MIC_RATE * 2)  # ...but the utterance began 2 s ago
    ctl._on_utterance(two_seconds, "voice")
    assert ctl.jobs.get(timeout=1)[3] is True


def test_clips_without_a_human_voice_are_not_transcribed(events):
    ctl, stt, *_ = make_controller("natural")
    heard: list = []
    ctl.speech = lambda audio: heard.append(len(audio)) or 0  # fan noise
    stt.texts = ["I love that. Thank you."]
    ctl.jobs.put(("final", bytes(vh.FRAME_BYTES * 40), "voice", False, None))
    wait_until(lambda: heard and ctl.jobs.empty())
    time.sleep(0.05)
    assert stt.calls == 0 and events.of("transcript") == []
    # push-to-talk is never second-guessed
    ctl.jobs.put(("final", bytes(vh.FRAME_BYTES * 40), "ptt", False, None))
    events.wait_for("transcript")
    assert len(heard) == 1


def test_over_our_own_voice_more_speech_is_needed(events):
    ctl, stt, *_ = make_controller("natural")
    ctl.speech = lambda audio: 450  # between speech_min_ms and speech_min_overlap_ms
    stt.texts = ["could you check the logs", "could you check the logs"]
    ctl.jobs.put(("final", bytes(vh.FRAME_BYTES * 40), "voice", True, None))
    ctl.jobs.put(("final", bytes(vh.FRAME_BYTES * 40), "voice", False, None))
    events.wait_for("transcript")
    time.sleep(0.05)
    assert stt.calls == 1 and len(events.of("transcript")) == 1


LOUD = (np.ones(vh.FRAME_SAMPLES, dtype=np.int16) * 6000).tobytes()


def barge_while_speaking(ctl):
    """Talk loudly over Claude until the barge-in check has answered."""
    lis = ctl.listener
    lis.vad = lambda f: vh.rms(f) > 0.01
    ctl.speaker.speaking = True  # mid-sentence, as far as the listener can tell
    for _ in range(30):
        lis.feed(LOUD)
    assert lis.verify_sent
    wait_until(lambda: lis.verdict is not None or not lis.unverified)
    lis.feed(LOUD)


def test_echo_while_speaking_does_not_stop_the_speaker(events):
    ctl, stt, _, player, _ = make_controller("natural", loudness_ratio=0)
    ctl.speaker.last_spoken = "The build passed and I pushed the branch."
    stt.texts = ["the build passed and I pushed"]
    barge_while_speaking(ctl)
    assert not events.of("barge_in") and player.stopped == 0
    assert not ctl.listener.in_speech


def test_empty_transcript_while_speaking_is_ignored(events):
    ctl, stt, _, player, _ = make_controller("natural", loudness_ratio=0)
    ctl.speaker.last_spoken = "The build passed and I pushed the branch."
    stt.texts = [""]
    barge_while_speaking(ctl)
    assert not events.of("barge_in") and player.stopped == 0
    assert not ctl.listener.in_speech


def talk_after_the_stop(ctl, frames=20):
    """Drop through the echo tail, then say something and pause."""
    lis = ctl.listener
    for _ in range(vh.BARGE_IN_TAIL_MS // vh.FRAME_MS):
        lis.feed(LOUD)  # the speaker's echo tail: dropped
    for _ in range(frames):
        lis.feed(LOUD)
    for _ in range(20):
        lis.feed(silent_frame())


def test_real_words_while_speaking_stop_the_speaker_but_only_what_follows_is_delivered(events):
    ctl, stt, _, player, _ = make_controller("natural", loudness_ratio=0, silence_ms=300,
                                             min_utterance_ms=200)
    heard: list[int] = []
    transcribe = stt.transcribe
    stt.transcribe = lambda audio: heard.append(len(audio)) or transcribe(audio)
    ctl.speaker.last_spoken = "The build passed and I pushed the branch."
    stt.texts = ["Because of five interviews."] + ["Use the other database."] * 3
    barge_while_speaking(ctl)
    assert events.of("barge_in") and player.stopped >= 1
    assert not ctl.listener.in_speech  # the overlapped audio is gone
    ctl.speaker.speaking = False
    talk_after_the_stop(ctl)
    events.wait_for("transcript")
    assert [e["text"] for e in events.of("transcript")] == ["Use the other database."]
    assert len(events.of("barge_in")) == 1
    # only post-stop audio (and silence) reached the recognizer for the turn
    assert max(heard[1:]) <= (20 + 20) * vh.FRAME_SAMPLES


def test_nothing_said_after_a_barge_in_sends_nothing(events):
    ctl, stt, _, player, _ = make_controller("natural", loudness_ratio=0, silence_ms=300,
                                             min_utterance_ms=200)
    ctl.speaker.last_spoken = "The build passed and I pushed the branch."
    stt.texts = ["Stop."] + ["Because of five interviews."] * 3
    barge_while_speaking(ctl)
    assert events.of("barge_in") and player.stopped >= 1
    ctl.speaker.speaking = False
    for _ in range((vh.BARGE_IN_TAIL_MS + vh.BARGE_IN_FOLLOW_UP_MS) // vh.FRAME_MS + 20):
        ctl.listener.feed(silent_frame())
    time.sleep(0.1)
    assert events.of("transcript") == [] and stt.calls == 1
    assert ctl.listener.follow_up == 0


def test_listener_drops_the_overlap_and_the_echo_tail_after_a_barge_in():
    checks: list = []
    lis, got, barge = make_listener(speaking=lambda: True, checks=checks)
    for _ in range(30):
        lis.feed(LOUD)
    lis.verify_result(checks[0][1], True)
    lis.feed(LOUD)
    assert barge == [1] and not lis.in_speech
    for _ in range(vh.BARGE_IN_TAIL_MS // vh.FRAME_MS):
        lis.feed(LOUD)
    assert not lis.in_speech  # the echo tail is not heard at all
    for _ in range(20):
        lis.feed(LOUD)
    for _ in range(20):
        lis.feed(silent_frame())
    # one fresh utterance of post-stop audio only, though still "speaking"
    assert len(got) == 1 and got[0][1] == "voice"
    assert got[0][0] <= (20 + 20) * vh.FRAME_BYTES
    assert len(checks) == 1 and barge == [1]


def test_a_barge_in_that_ends_before_its_check_is_only_a_trigger(events):
    ctl, *_ = make_controller("natural")
    ctl.speaker.last_spoken = "The build passed and I pushed the branch."
    ctl.speaker.speaking = True
    ctl.handle_text("Because of five interviews.", "barge", during_speech=True)
    assert events.of("barge_in") and events.of("transcript") == []
    assert ctl.listener.restart_pending
    # Claude already finished: nothing to stop, so it is the person's turn
    ctl.speaker.speaking = False
    ctl.handle_text("Use the other database.", "barge", during_speech=True)
    assert events.of("transcript")[-1] == {"type": "transcript", "text": "Use the other database.",
                                           "source": "voice"}


def test_listener_marks_an_unchecked_barge_in(events):
    lis, got, barge = make_listener(speaking=lambda: True, checks=[])
    for _ in range(20):
        lis.feed(LOUD)
    for _ in range(20):
        lis.feed(silent_frame())
    assert got and got[-1][1] == "barge" and barge == []


def test_push_to_talk_right_after_a_barge_in_hears_everything():
    checks: list = []
    lis, got, barge = make_listener(speaking=lambda: True, checks=checks)
    for _ in range(30):
        lis.feed(LOUD)
    lis.verify_result(checks[0][1], True)
    lis.feed(LOUD)
    lis.ptt_down()
    for _ in range(20):
        lis.feed(LOUD)
    lis.ptt_up()
    assert got == [(20 * vh.FRAME_BYTES, "ptt", None)]
    assert lis.tail_skip == 0 and lis.follow_up == 0


def test_a_single_word_over_our_voice_waits_for_more_before_barging_in(events):
    ctl, stt, _, player, _ = make_controller("natural", loudness_ratio=0)
    ctl.speech = lambda audio: len(audio) * 1000 // vh.MIC_RATE  # all of it a voice
    ctl.speaker.last_spoken = "Here is a longer answer to test the fix."
    stt.texts = ["Important."]
    barge_while_speaking(ctl)
    assert not events.of("barge_in") and player.stopped == 0
    lis = ctl.listener
    assert lis.in_speech and lis.unverified  # still listening, not dropped
    stt.texts = ["Wait, use the other database."]
    for _ in range(30):
        lis.feed(LOUD)
    wait_until(lambda: lis.verdict is not None or not lis.unverified)
    lis.feed(LOUD)
    assert events.of("barge_in") and player.stopped >= 1


def test_echo_that_never_turns_into_words_is_dropped_after_a_few_checks(events):
    ctl, stt, _, player, _ = make_controller("natural", loudness_ratio=0)
    ctl.speech = lambda audio: 0  # the echo canceller's leftovers: no human voice
    lis = ctl.listener
    barge_while_speaking(ctl)
    for _ in range(vh.BARGE_IN_CHECKS):
        for _ in range(vh.BARGE_IN_VERIFY_MS // vh.FRAME_MS + 1):
            lis.feed(LOUD)
            if not lis.in_speech:
                break
        wait_until(lambda: lis.verdict is not None or not lis.unverified or not lis.in_speech)
        lis.feed(LOUD)
    assert not events.of("barge_in") and player.stopped == 0 and stt.calls == 0
    assert not lis.unverified


def test_a_real_interruption_passes_the_speech_check(events):
    ctl, stt, _, player, _ = make_controller("natural", loudness_ratio=0, silence_ms=300,
                                             min_utterance_ms=200)
    ctl.speech = lambda audio: len(audio) * 1000 // vh.MIC_RATE * 3 // 4
    ctl.speaker.last_spoken = "The build passed and I pushed the branch."
    stt.texts = ["Wait, use the other database."] + ["Use the other database."] * 3
    barge_while_speaking(ctl)
    assert events.of("barge_in") and player.stopped >= 1
    talk_after_the_stop(ctl)
    events.wait_for("transcript")
    assert events.of("transcript")[-1]["text"] == "Use the other database."


def test_wake_word_while_speaking_interrupts(events):
    ctl, _, _, player, _ = make_controller("wake")
    ctl.handle_text("Hey Claude stop", "voice", during_speech=True)
    assert events.of("barge_in")
    assert player.stopped >= 1
    assert events.of("transcript")[-1]["text"] == "stop"


def test_mode_and_mute(events):
    ctl, *_ = make_controller("wake")
    ctl.set_mode("ptt")
    assert ctl.listener.hands_free is False
    ctl.set_mode("natural")
    assert ctl.listener.hands_free is True
    ctl.set_muted(True)
    assert ctl.listener.hands_free is False
    assert events.of("state")[-1]["state"] == "muted"
    with pytest.raises(ValueError):
        ctl.set_mode("loud")


def test_ptt_down_interrupts_speech(events):
    ctl, _, tts, player, chimes = make_controller("natural")
    tts.seconds = 2.0
    ctl.speaker.say("a long sentence")
    wait_until(lambda: ctl.speaker.speaking)
    ctl.ptt("down")
    assert events.of("barge_in") and player.stopped >= 1
    assert "ptt_down" in chimes
    ctl.ptt("up")


# --------------------------------------------------------------------------
# spoken yes/no questions


def answer_later(ctl, text, delay=0.2):
    def run():
        wait_until(lambda: ctl.ask is not None and ctl.ask.listening)
        time.sleep(delay)
        ctl.handle_text(text, "voice")

    threading.Thread(target=run, daemon=True).start()


@pytest.mark.parametrize("mode", ["wake", "natural", "ptt"])
def test_ask_yes(events, mode):
    ctl, _, tts, _, _ = make_controller(mode)
    answer_later(ctl, "yeah go ahead")
    result = ctl.ask_question("The agent wants to run npm test. Allow it?", timeout=5)
    assert result == {"answer": "yes", "heard": "yeah go ahead"}
    assert tts.said[0].startswith("The agent wants")
    # the answer is not also sent to Claude as a message
    assert events.of("transcript") == []
    assert ctl.ask is None


def test_ask_no_with_wake_word(events):
    ctl, *_ = make_controller("wake")
    answer_later(ctl, "Hey Claude, no don't")
    assert ctl.ask_question("Allow?", timeout=5)["answer"] == "no"


def test_ask_unclear_asks_again_once(events):
    ctl, _, tts, _, _ = make_controller("natural")

    def run():
        wait_until(lambda: ctl.ask is not None and ctl.ask.listening)
        ctl.handle_text("what is it doing", "voice")
        wait_until(lambda: "Sorry, was that a yes or a no?" in tts.said)
        wait_until(lambda: ctl.ask.listening)
        ctl.handle_text("yes", "voice")

    threading.Thread(target=run, daemon=True).start()
    assert ctl.ask_question("Allow?", timeout=5)["answer"] == "yes"


def test_ask_times_out(events):
    ctl, *_ = make_controller("natural")
    assert ctl.ask_question("Allow?", timeout=0.3)["answer"] == "timeout"
    assert ctl.ask is None


def test_ptt_mode_listens_hands_free_only_while_asking(events):
    ctl, *_ = make_controller("ptt")
    seen = []

    def run():
        wait_until(lambda: ctl.ask is not None and ctl.ask.listening)
        seen.append(ctl.listener.hands_free)
        ctl.handle_text("yes", "voice")

    threading.Thread(target=run, daemon=True).start()
    ctl.ask_question("Allow?", timeout=5)
    assert seen == [True]
    assert ctl.listener.hands_free is False


# --------------------------------------------------------------------------
# control socket


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("voice", timeout=10)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.path)


def request(path, method, url, body=None):
    conn = UnixHTTPConnection(path)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, url, body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = json.loads(resp.read() or b"{}")
    conn.close()
    return resp.status, out


@pytest.fixture
def server(events):
    ctl, stt, tts, player, _ = make_controller("wake")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "voice.sock")
    stopped = threading.Event()
    srv = vh._UnixHTTPServer(path, vh.make_handler(ctl, stopped.set))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield path, ctl, tts, stopped
    srv.shutdown()
    srv.server_close()


def test_socket_speak_mode_ptt_status(server):
    path, ctl, tts, _ = server
    assert request(path, "POST", "/speak", {"text": "Hello there."})[0] == 200
    wait_until(lambda: tts.said == ["Hello there."])
    assert request(path, "POST", "/mode", {"mode": "natural"})[1]["mode"] == "natural"
    assert request(path, "POST", "/mode", {"mode": "nope"})[0] == 400
    assert request(path, "POST", "/ptt/toggle")[1]["ptt"] is True
    assert request(path, "POST", "/ptt/toggle")[1]["ptt"] is False
    status, snap = request(path, "GET", "/status")
    assert status == 200 and snap["mode"] == "natural"
    assert request(path, "POST", "/nope")[0] == 404


def test_socket_ask_blocks_until_answered(server):
    path, ctl, _, _ = server
    answer_later(ctl, "nope")
    status, out = request(path, "POST", "/ask", {"question": "Allow?", "timeout": 5})
    assert status == 200 and out["answer"] == "no"


def test_socket_shutdown(server):
    path, _, _, stopped = server
    assert request(path, "POST", "/shutdown")[0] == 200
    assert stopped.wait(2)


def test_take_over_socket_stops_the_other_helper(server):
    path, _, _, stopped = server
    vh.take_over_socket(path)
    assert stopped.wait(2)
    assert not os.path.exists(path)


# --------------------------------------------------------------------------
# main(): the whole process with fake engines and fake audio commands


def test_main_end_to_end(events, monkeypatch):
    stt = ScriptedSTT()
    stt.texts = ["Hey Claude, what's the plan?"]
    monkeypatch.setattr(vh, "make_stt", lambda cfg: setattr(stt, "device", "cpu") or stt)
    monkeypatch.setattr(vh, "PhononTTS", lambda cfg: FakeTTS())

    class Voice:  # the speech check, without downloading its model
        speech_ms = staticmethod(lambda audio: len(audio) * 1000 // vh.MIC_RATE)

    monkeypatch.setattr(vh, "SileroVAD", Voice)
    tmp = tempfile.mkdtemp()
    sock = os.path.join(tmp, "v.sock")
    # The "mic": 0.6 s silence, 1 s of loud noise (speech to the VAD), 1 s silence, then idle.
    mic = (
        "import sys,time,random\n"
        "w=sys.stdout.buffer\n"
        "w.write(bytes(960*20))\n"
        "import struct\n"
        "for _ in range(33): w.write(b''.join(struct.pack('<h', random.randint(-12000,12000)) for _ in range(480)))\n"
        "w.write(bytes(960*40)); w.flush(); time.sleep(30)\n"
    )
    cfg = {
        "mode": "wake",
        "capture_cmd": [sys.executable, "-c", mic],
        "playback_cmd": ["sh", "-c", "cat > /dev/null"],
        "vad_aggressiveness": 0,
        "chimes": False,
    }
    rc: list[int] = []
    t = threading.Thread(target=lambda: rc.append(vh.main(["--socket", sock, "--config", json.dumps(cfg)])), daemon=True)
    monkeypatch.setattr(vh.signal, "signal", lambda *a: None)  # not on the main thread
    t.start()
    ready = events.wait_for("ready", timeout=10)[0]
    assert ready["socket"] == sock and ready["mode"] == "wake"
    transcript = events.wait_for("transcript", timeout=10)[0]
    assert transcript["text"] == "what's the plan"
    assert request(sock, "POST", "/shutdown")[0] == 200
    t.join(5)
    assert rc == [0]
    assert not os.path.exists(sock)


# --------------------------------------------------------------------------
# end-of-turn detection


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Can you run the tests?", False),
        ("Fix the login bug.", False),
        ("Can you check whether the", True),
        ("So I was thinking we could, um", True),
        ("refactor the handler and", True),
        ("I want to,", True),
        ("Let me think...", True),
        ("", False),
    ],
)
def test_ends_mid_sentence(text, expected):
    assert vh.ends_mid_sentence(text) is expected


def make_turn_listener(turn, check_text=False, **cfg_overrides):
    got: list = []
    pauses: list = []
    cfg_overrides.setdefault("loudness_ratio", 0)
    cfg = vh.Config(mode="natural", silence_ms=300, early_ms=90, min_utterance_ms=200,
                    turn_max_silence_ms=900, **cfg_overrides)
    lis = vh.Listener(
        cfg,
        vad=lambda f: vh.rms(f) > 0.01,
        on_utterance=lambda audio, src, key: got.append((len(audio), src, key)),
        is_speaking=lambda: False,
        on_barge_in=lambda: None,
        on_hearing=lambda h: None,
        on_pause=lambda audio, key: pauses.append(key),
        turn=turn,
        check_text=check_text,
    )
    lis.turn_sync = True
    return lis, got, pauses


def silent_frames_until_utterance(lis, got, limit=60):
    """Feed silence until an utterance ends; return how many frames it took."""
    for n in range(1, limit + 1):
        lis.feed(silent_frame())
        if got:
            return n
    return None


def test_turn_complete_ends_the_utterance_at_the_short_pause(events):
    seen: list = []
    lis, got, pauses = make_turn_listener(lambda audio: seen.append(len(audio)) or 0.95)
    for _ in range(30):
        lis.feed(voiced_frame())
    # early_ms=90: three silent frames, not the ten that silence_ms=300 needs
    assert silent_frames_until_utterance(lis, got) == 3
    assert len(seen) == 1 and seen[0] > 0
    assert got[0][2] == pauses[0]  # the early transcript is reused
    assert any("turn complete" in e["msg"] for e in events.of("log"))


def test_turn_incomplete_keeps_listening_through_a_long_pause():
    lis, got, _ = make_turn_listener(lambda audio: 0.05)
    for _ in range(30):
        lis.feed(voiced_frame())
    for _ in range(20):  # 600 ms: past silence_ms, short of turn_max_silence_ms
        lis.feed(silent_frame())
    assert got == []
    for _ in range(20):  # they go on
        lis.feed(voiced_frame())
    assert got == []
    # The next pause is judged afresh; still unfinished, it ends at 900 ms.
    assert silent_frames_until_utterance(lis, got) == 30
    assert len(got) == 1


def test_turn_detection_off_uses_the_fixed_pause():
    lis, got, _ = make_turn_listener(None)
    for _ in range(30):
        lis.feed(voiced_frame())
    assert silent_frames_until_utterance(lis, got) == 10  # silence_ms=300


def test_turn_model_failure_falls_back_to_the_fixed_pause(events):
    def broken(audio):
        raise RuntimeError("boom")

    lis, got, _ = make_turn_listener(broken)
    for _ in range(30):
        lis.feed(voiced_frame())
    assert silent_frames_until_utterance(lis, got) == 10
    assert any("turn detection failed" in e["msg"] for e in events.of("log"))


def test_turn_waits_for_a_model_that_has_not_answered():
    lis, got, _ = make_turn_listener(lambda audio: 0.95)
    lis.turn_sync = False
    gate = threading.Event()
    lis.turn = lambda audio: gate.wait(5) and 0.95
    for _ in range(30):
        lis.feed(voiced_frame())
    for _ in range(15):  # past silence_ms with no answer yet: don't cut in
        lis.feed(silent_frame())
    assert got == []
    gate.set()
    wait_until(lambda: lis.turn_result is not None)
    lis.feed(silent_frame())
    assert len(got) == 1


def test_turn_complete_but_dangling_transcript_keeps_listening():
    lis, got, pauses = make_turn_listener(lambda audio: 0.95, check_text=True)
    for _ in range(30):
        lis.feed(voiced_frame())
    for _ in range(3):
        lis.feed(silent_frame())
    lis.turn_text(pauses[0], "Can you check whether the")
    for _ in range(20):  # 690 ms of silence in all: still waiting
        lis.feed(silent_frame())
    assert got == []
    assert silent_frames_until_utterance(lis, got) == 7  # 900 ms


def test_turn_complete_ends_as_soon_as_the_transcript_agrees():
    lis, got, pauses = make_turn_listener(lambda audio: 0.95, check_text=True)
    for _ in range(30):
        lis.feed(voiced_frame())
    for _ in range(4):
        lis.feed(silent_frame())
    assert got == []  # the transcript is not in yet
    lis.turn_text(pauses[0], "Run the tests.")
    assert silent_frames_until_utterance(lis, got) == 1


def test_turn_complete_without_a_transcript_ends_after_the_usual_pause():
    lis, got, _ = make_turn_listener(lambda audio: 0.95, check_text=True)
    for _ in range(30):
        lis.feed(voiced_frame())
    assert silent_frames_until_utterance(lis, got) == 10


def test_controller_passes_the_early_transcript_to_the_turn_check(events):
    cfg = vh.Config(mode="natural")
    stt = ScriptedSTT()
    stt.texts = ["Can you check whether the"]
    ctl = vh.Controller(cfg, stt, lambda c: vh.Speaker(FakeTTS(), FakePlayer(), c.on_speaking),
                        turn=lambda audio: 0.9)
    assert ctl.listener.turn is not None and ctl.listener.check_text
    ctl._on_pause(b"\0\0" * 1600, (1, 40))
    wait_until(lambda: ctl.listener.text_result == ((1, 40), False))


def test_smart_turn_features_shape():
    feats = vh.whisper_features(np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1)
    assert feats.shape == (80, 800) and feats.dtype == np.float32
    # front-padded: the first 7 s are the same (silent) column throughout
    assert np.allclose(feats[:, :600], feats[:, :1])
    assert not np.allclose(feats[:, -50:], feats[:, :1])


def test_main_falls_back_when_the_turn_model_will_not_load(events, monkeypatch):
    def no_model(cfg):
        raise RuntimeError("no network")

    monkeypatch.setattr(vh, "SmartTurn", no_model)
    test_main_end_to_end(events, monkeypatch)
    assert any("turn detection unavailable" in e["msg"] for e in events.of("log"))
