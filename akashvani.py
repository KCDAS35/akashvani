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
import time

import sounddevice as sd
import numpy as np
import pyperclip
import ctypes
from faster_whisper import WhisperModel
from pynput import keyboard

try:
    import tkinter as tk
    HAVE_TK = True
except Exception:
    HAVE_TK = False

# ── Config ────────────────────────────────────────────────────────────────────
TRIGGER_KEY   = keyboard.Key.f12   # Hold to record, release to transcribe
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")  # tiny.en / base.en / small.en
SAMPLE_RATE   = 16000
CHANNELS      = 1
CHUNK         = 1024
MIN_DURATION    = 0.3   # seconds — ignore accidental taps shorter than this
TOGGLE_DEBOUNCE = 0.3   # seconds — ignore rapid presses (X11 auto-repeat)

# ── State ─────────────────────────────────────────────────────────────────────
recording        = False
audio_frames     = []
record_lock      = threading.Lock()
model            = None
last_toggle_time: float = 0.0
mic_ui           = None

# ── Audio helpers ─────────────────────────────────────────────────────────────

def start_recording():
    global recording, audio_frames
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


def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_frames.append(indata.copy())


def audio_loop():
    """Background thread: open mic stream and keep it alive."""
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", blocksize=CHUNK,
                        callback=audio_callback):
        while True:
            time.sleep(0.1)


def save_wav(frames) -> str:
    """Write captured numpy frames to a temp WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_data = np.concatenate(frames, axis=0)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
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
    """Copy to clipboard then paste with Ctrl+V — works in every Windows app."""
    if not text:
        return
    pyperclip.copy(text)
    # Simulate Ctrl+V via Windows SendInput
    VK_CONTROL, VK_V = 0x11, 0x56
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT)]
        _anonymous_ = ("_input",)
        _fields_ = [("type", ctypes.c_ulong), ("_input", _INPUT)]

    def make_key(vk, flags=0):
        i = INPUT()
        i.type = INPUT_KEYBOARD
        i.ki.wVk = vk
        i.ki.dwFlags = flags
        return i

    inputs = [
        make_key(VK_CONTROL),
        make_key(VK_V),
        make_key(VK_V, KEYEVENTF_KEYUP),
        make_key(VK_CONTROL, KEYEVENTF_KEYUP),
    ]
    arr = (INPUT * len(inputs))(*inputs)
    ctypes.windll.user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def notify(msg: str):
    """Desktop notification via Windows balloon tooltip (best-effort)."""
    try:
        ctypes.windll.user32.MessageBeep(0)
        # Use a non-blocking thread to show a Windows toast via PowerShell
        cmd = (
            f'powershell -WindowStyle Hidden -Command "'
            f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;'
            f'$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText01);'
            f'$t.GetElementsByTagName(\'text\')[0].AppendChild($t.CreateTextNode(\'{msg[:80].replace(chr(39), "")}\')) | Out-Null;'
            f'[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier(\'Akash Vani\').Show([Windows.UI.Notifications.ToastNotification]::new($t))'
            f'"'
        )
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── Toggle (shared by F12 and floating-mic click) ─────────────────────────────

press_time: float = 0.0


def _ui_state(state: str):
    if mic_ui is not None:
        mic_ui.root.after(0, mic_ui.set_state, state)


def toggle():
    global press_time, last_toggle_time
    now = time.monotonic()
    if now - last_toggle_time < TOGGLE_DEBOUNCE:
        return
    last_toggle_time = now

    if not recording:
        press_time = now
        start_recording()
        _ui_state("recording")
    else:
        duration = now - press_time
        frames_snapshot = list(audio_frames)
        stop_recording()
        if duration < MIN_DURATION or not frames_snapshot:
            print(f"⚡ Too short, ignoring. (duration={duration:.2f}s, frames={len(frames_snapshot)})", flush=True)
            _ui_state("idle")
            return
        _ui_state("thinking")
        threading.Thread(
            target=process_audio,
            args=(frames_snapshot,),
            daemon=True,
        ).start()


def on_press(key):
    if key != TRIGGER_KEY:
        return
    toggle()


# ── Floating mic UI ───────────────────────────────────────────────────────────

class FloatingMic:
    """Always-on-top draggable mic button. Click toggles recording."""

    SIZE = 80
    MARGIN = 30

    STATES = {
        "idle":      {"fill": "#1e1e1e", "ring": "#555555", "text": "🎙"},
        "recording": {"fill": "#c81e1e", "ring": "#ff6464", "text": "⏹"},
        "thinking":  {"fill": "#1e64c8", "ring": "#64a0ff", "text": "💬"},
    }

    def __init__(self, on_click):
        self.root = tk.Tk()
        self.root.title("akashvani")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.92)
        except Exception:
            pass

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = sw - self.SIZE - self.MARGIN
        y = sh - self.SIZE - self.MARGIN
        self.root.geometry(f"{self.SIZE}x{self.SIZE}+{x}+{y}")

        bg = self.STATES["idle"]["fill"]
        self.canvas = tk.Canvas(
            self.root, width=self.SIZE, height=self.SIZE,
            highlightthickness=0, bg=bg,
        )
        self.canvas.pack(fill="both", expand=True)

        pad = 2
        self.disc = self.canvas.create_oval(
            pad, pad, self.SIZE - pad, self.SIZE - pad,
            fill=bg, outline=self.STATES["idle"]["ring"], width=2,
        )
        self.label = self.canvas.create_text(
            self.SIZE // 2, self.SIZE // 2,
            text=self.STATES["idle"]["text"],
            fill="#ffffff", font=("Sans", 28),
        )

        for tag in (self.canvas, "<Button-1>", "<B1-Motion>", "<ButtonRelease-1>"):
            pass
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)

        self._on_click = on_click
        self._dragging = False
        self._origin = (0, 0)
        self._win_origin = (0, 0)

    def _press(self, e):
        self._dragging = False
        self._origin = (e.x_root, e.y_root)
        self._win_origin = (self.root.winfo_x(), self.root.winfo_y())

    def _drag(self, e):
        dx = e.x_root - self._origin[0]
        dy = e.y_root - self._origin[1]
        if abs(dx) > 4 or abs(dy) > 4:
            self._dragging = True
        self.root.geometry(f"+{self._win_origin[0] + dx}+{self._win_origin[1] + dy}")

    def _release(self, e):
        if not self._dragging:
            self._on_click()

    def set_state(self, state: str):
        s = self.STATES.get(state, self.STATES["idle"])
        self.canvas.config(bg=s["fill"])
        self.canvas.itemconfig(self.disc, fill=s["fill"], outline=s["ring"])
        self.canvas.itemconfig(self.label, text=s["text"])

    def run(self):
        self.root.mainloop()


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
        _ui_state("idle")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global model, mic_ui

    print(f"🌌 Akash Vani — loading model '{WHISPER_MODEL}' ...", flush=True)
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("✅ Model ready. F12 or click the floating mic. Ctrl+C to quit.\n", flush=True)

    threading.Thread(target=audio_loop, daemon=True).start()
    keyboard.Listener(on_press=on_press).start()

    if HAVE_TK and os.environ.get("DISPLAY"):
        try:
            mic_ui = FloatingMic(on_click=toggle)
            mic_ui.run()
            return
        except Exception as e:
            print(f"⚠ Floating mic failed ({e}); falling back to F12-only.", flush=True)

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n👋 Bye!", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
