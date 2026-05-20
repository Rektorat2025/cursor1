from __future__ import annotations

import tempfile
import threading
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


LANGUAGES = {
    "Auto detect Polish or English": None,
    "English": "en",
    "Polish": "pl",
}

MODEL_SIZES = ("tiny", "base", "small", "medium")
DEFAULT_MODEL_SIZE = "base"


def write_wav(path: Path, chunks: Iterable[np.ndarray], sample_rate: int) -> None:
    """Write float32 mono audio chunks to a PCM WAV file."""
    audio_parts = [chunk.reshape(-1, 1) for chunk in chunks if chunk.size]
    if not audio_parts:
        raise ValueError("No audio was captured.")

    audio = np.concatenate(audio_parts, axis=0)
    audio = np.clip(audio, -1.0, 1.0)
    pcm_audio = (audio * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio.tobytes())


class MicTranscriberApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mic Transcriber")
        self.root.geometry("780x560")
        self.root.minsize(640, 440)

        self.language_var = tk.StringVar(value="Auto detect Polish or English")
        self.model_size_var = tk.StringVar(value=DEFAULT_MODEL_SIZE)
        self.status_var = tk.StringVar(value="Ready.")

        self._audio_chunks: list[np.ndarray] = []
        self._audio_lock = threading.Lock()
        self._is_recording = False
        self._stream: sd.InputStream | None = None
        self._sample_rate = 16_000
        self._model = None
        self._model_size: str | None = None
        self._model_lock = threading.Lock()

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(4, weight=1)

        title = ttk.Label(
            container,
            text="Microphone transcription",
            font=("TkDefaultFont", 16, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        help_text = (
            "Record speech from your microphone, then transcribe it locally with Whisper. "
            "Choose automatic language detection or lock transcription to English or Polish."
        )
        ttk.Label(container, text=help_text, wraplength=720).grid(
            row=1, column=0, sticky="ew", pady=(6, 14)
        )

        controls = ttk.Frame(container)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="Language:").grid(row=0, column=0, sticky="w")
        self.language_combo = ttk.Combobox(
            controls,
            textvariable=self.language_var,
            values=tuple(LANGUAGES.keys()),
            state="readonly",
            width=30,
        )
        self.language_combo.grid(row=0, column=1, sticky="w", padx=(8, 18))

        ttk.Label(controls, text="Model:").grid(row=0, column=2, sticky="w")
        self.model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_size_var,
            values=MODEL_SIZES,
            state="readonly",
            width=10,
        )
        self.model_combo.grid(row=0, column=3, sticky="w", padx=(8, 0))

        buttons = ttk.Frame(container)
        buttons.grid(row=3, column=0, sticky="ew", pady=14)

        self.start_button = ttk.Button(buttons, text="Start recording", command=self.start_recording)
        self.start_button.grid(row=0, column=0, padx=(0, 8))

        self.stop_button = ttk.Button(
            buttons,
            text="Stop and transcribe",
            command=self.stop_recording,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, padx=(0, 8))

        ttk.Button(buttons, text="Save transcript", command=self.save_transcript).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(buttons, text="Clear", command=self.clear_transcript).grid(row=0, column=3)

        self.transcript_text = scrolledtext.ScrolledText(container, wrap="word", undo=True)
        self.transcript_text.grid(row=4, column=0, sticky="nsew")

        status = ttk.Label(container, textvariable=self.status_var, anchor="w")
        status.grid(row=5, column=0, sticky="ew", pady=(10, 0))

    def start_recording(self) -> None:
        if self._is_recording:
            return

        try:
            self._sample_rate = self._default_input_sample_rate()
            with self._audio_lock:
                self._audio_chunks = []

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._record_audio,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            messagebox.showerror("Microphone error", f"Could not start microphone recording:\n\n{exc}")
            self.status_var.set("Microphone is not available.")
            return

        self._is_recording = True
        self._set_controls(recording=True, transcribing=False)
        self.status_var.set("Recording... speak Polish or English, then click Stop and transcribe.")

    def stop_recording(self) -> None:
        if not self._is_recording:
            return

        self._is_recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

        with self._audio_lock:
            chunks = list(self._audio_chunks)
            self._audio_chunks = []

        if not chunks:
            self._set_controls(recording=False, transcribing=False)
            self.status_var.set("No audio was captured.")
            return

        language_code = LANGUAGES[self.language_var.get()]
        model_size = self.model_size_var.get()
        self._set_controls(recording=False, transcribing=True)
        self.status_var.set("Transcribing... the first run may download the selected Whisper model.")

        worker = threading.Thread(
            target=self._transcribe_chunks,
            args=(chunks, self._sample_rate, language_code, model_size),
            daemon=True,
        )
        worker.start()

    def save_transcript(self) -> None:
        transcript = self.transcript_text.get("1.0", "end").strip()
        if not transcript:
            messagebox.showinfo("Nothing to save", "There is no transcript text to save yet.")
            return

        path = filedialog.asksaveasfilename(
            title="Save transcript",
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
        )
        if not path:
            return

        try:
            Path(path).write_text(transcript + "\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save transcript:\n\n{exc}")
            return

        self.status_var.set(f"Saved transcript to {path}")

    def clear_transcript(self) -> None:
        self.transcript_text.delete("1.0", "end")
        self.status_var.set("Transcript cleared.")

    def _record_audio(self, indata: np.ndarray, _frames: int, _time_info: object, status: object) -> None:
        if status:
            self.root.after(0, self.status_var.set, f"Recording warning: {status}")

        if self._is_recording:
            with self._audio_lock:
                self._audio_chunks.append(indata.copy())

    def _transcribe_chunks(
        self,
        chunks: list[np.ndarray],
        sample_rate: int,
        language_code: str | None,
        model_size: str,
    ) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = Path(temp_audio.name)

            write_wav(temp_path, chunks, sample_rate)
            model = self._get_model(model_size)

            segments, info = model.transcribe(
                str(temp_path),
                language=language_code,
                task="transcribe",
                beam_size=5,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()

            if not text:
                text = "(No speech detected.)"

            detected_language = getattr(info, "language", None)
            language_probability = getattr(info, "language_probability", None)
            self.root.after(
                0,
                self._finish_transcription,
                text,
                detected_language,
                language_probability,
            )
        except Exception as exc:
            self.root.after(0, self._fail_transcription, exc)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _get_model(self, model_size: str):
        with self._model_lock:
            if self._model is not None and self._model_size == model_size:
                return self._model

            from faster_whisper import WhisperModel

            self.root.after(0, self.status_var.set, f"Loading Whisper '{model_size}' model...")
            self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self._model_size = model_size
            return self._model

    def _finish_transcription(
        self,
        text: str,
        detected_language: str | None,
        language_probability: float | None,
    ) -> None:
        prefix = ""
        if detected_language:
            prefix = f"[{detected_language}"
            if language_probability is not None:
                prefix += f" {language_probability:.0%}"
            prefix += "] "

        if self.transcript_text.get("1.0", "end").strip():
            self.transcript_text.insert("end", "\n\n")
        self.transcript_text.insert("end", prefix + text)
        self.transcript_text.see("end")

        self._set_controls(recording=False, transcribing=False)
        self.status_var.set("Transcription complete.")

    def _fail_transcription(self, exc: Exception) -> None:
        self._set_controls(recording=False, transcribing=False)
        self.status_var.set("Transcription failed.")
        messagebox.showerror("Transcription failed", str(exc))

    def _set_controls(self, recording: bool, transcribing: bool) -> None:
        if recording:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.language_combo.configure(state="disabled")
            self.model_combo.configure(state="disabled")
            return

        if transcribing:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.language_combo.configure(state="disabled")
            self.model_combo.configure(state="disabled")
            return

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.language_combo.configure(state="readonly")
        self.model_combo.configure(state="readonly")

    @staticmethod
    def _default_input_sample_rate() -> int:
        device = sd.query_devices(kind="input")
        sample_rate = device.get("default_samplerate", 16_000)
        return int(sample_rate)


def main() -> None:
    root = tk.Tk()
    ttk.Style(root)
    MicTranscriberApp(root)
    root.mainloop()
