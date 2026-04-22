#!/usr/bin/env python3
"""
Akash Vani — Voice from the Sky.
Press F12 to start recording, press again to stop and transcribe.
Uses faster-whisper for local, offline speech-to-text.
"""

import os
import sys
import wave
import tempfile
import subprocess
import threading
import queue
import time

import pyaudio
import pyperclip
from faster_whisper import WhisperModel
from pynput import keyboard

# ── Config ────────────────────────────────────────────────────────────────────
TRIGGER_KEY   = keyboard.Key.f12   # Hold to record, release to transcribe
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")  # tiny.en / base.en / small.en
SAMPLE_RATE   = 16000
CHANNELS      = 1
CHUNK         = 1024
AUDIO_FORMAT  = pyaudio.paInt16
MIN_DURATION    = 0.3   # seconds — ignore accidental taps shorter than this
TOGGLE_DEBOUNCE = 0.3   # seconds — ignore rapid presses (X11 auto-repeat)

# ── State ─────────────────────────────────────────────────────────────────────
recording        = False
audio_frames     = []
record_lock      = threading.Lock()
model            = None
pa               = None
last_toggle_time: float = 0.0

# ── Audio helpers ─────────────────────────────────────────────────────────────

def start_recording():
    global recording, audio_frames, stream
    with record_lock:
        if recording:
            return
        recording = True
        audio_frames = []
    print("🎙  Recording...", flush=True)


def stop_recording():
    global recording
    with record_lock:
        if not recording:
            return False
        recording = False
    print("⏹  Stopped.", flush=True)
    return True


def audio_loop():
    """Background thread: capture mic while recording=True."""
    global pa, audio_frames
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except Exception:
            time.sleep(0.01)
            continue
        if recording:
            audio_frames.append(data)


def save_wav(frames) -> str:
    """Write PCM frames to a temp WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return tmp.name


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(wav_path: str) -> str:
    segments, _ = model.transcribe(
        wav_path,
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segments).strip()


# ── Type text into active window ───────────────────────────────────────────────

def type_text(text: str):
    """Copy to clipboard then paste — works in every app including terminals."""
    if not text:
        return
    try:
        pyperclip.copy(text)
        subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
                       check=False)
    except Exception:
        # Fallback: xdotool type (slower but works when clipboard paste fails)
        subprocess.run(["xdotool", "type", "--clearmodifiers", "--", text],
                       check=False)


def notify(msg: str):
    """Desktop notification (best-effort)."""
    try:
        subprocess.run(["notify-send", "-t", "3000", "SoupaWhisper", msg],
                       check=False, capture_output=True)
    except Exception:
        pass


# ── Key listener ──────────────────────────────────────────────────────────────

press_time: float = 0.0


def on_press(key):
    global press_time, last_toggle_time
    if key != TRIGGER_KEY:
        return
    now = time.monotonic()
    if now - last_toggle_time < TOGGLE_DEBOUNCE:
        return  # auto-repeat or accidental double-tap — ignore
    last_toggle_time = now

    if not recording:
        press_time = now
        start_recording()
    else:
        duration = now - press_time
        frames_snapshot = list(audio_frames)
        stop_recording()
        if duration < MIN_DURATION or not frames_snapshot:
            print(f"⚡ Too short, ignoring. (duration={duration:.2f}s, frames={len(frames_snapshot)})", flush=True)
            return
        # Transcribe in a thread so the key listener stays responsive
        threading.Thread(
            target=process_audio,
            args=(frames_snapshot,),
            daemon=True,
        ).start()


def process_audio(frames):
    wav_path = save_wav(frames)
    try:
        print("🔍 Transcribing...", flush=True)
        text = transcribe(wav_path)
        if text:
            print(f"📝 {text}", flush=True)
            type_text(text)
            notify(text[:80] + ("…" if len(text) > 80 else ""))
        else:
            print("🔇 Nothing detected.", flush=True)
    finally:
        os.unlink(wav_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global model

    print(f"🌌 Akash Vani — loading model '{WHISPER_MODEL}' ...", flush=True)
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("✅ Model ready. Press F12 to start recording, press again to stop. Ctrl+C to quit.\n", flush=True)

    # Start audio capture thread
    t = threading.Thread(target=audio_loop, daemon=True)
    t.start()

    # Start key listener (blocking)
    with keyboard.Listener(on_press=on_press) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\n👋 Bye!", flush=True)
            sys.exit(0)


if __name__ == "__main__":
    main()
