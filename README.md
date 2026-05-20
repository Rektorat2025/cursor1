# Mic Transcriber

A small desktop app that records your microphone and transcribes speech to text.
It uses Whisper locally through `faster-whisper` and supports English, Polish, or
automatic language detection between the two.

## Features

- Simple desktop UI built with Tkinter.
- Records from the default system microphone.
- Transcribes locally after recording stops.
- Supports Polish (`pl`), English (`en`), and automatic detection.
- Saves transcripts as UTF-8 text files.
- Includes a PyInstaller build configuration for a standalone executable.

## Run from source

On Debian/Ubuntu, install the system packages needed by Tkinter, virtual
environments, and microphone input first:

```bash
sudo apt-get install python3.12-venv python3-tk libportaudio2
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mic-transcriber
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
mic-transcriber
```

The first transcription can take longer because the selected Whisper model is
downloaded and cached. After that, transcription runs locally from the cached
model. The default `base` model is a balanced choice for CPU-only machines;
`small` and `medium` are more accurate but slower and larger.

## Build a standalone executable

Install the build dependency, then run the build script:

```bash
pip install -r requirements.txt -r requirements-build.txt
python scripts/build_exe.py
```

PyInstaller writes the app to `dist/`:

- Windows: `dist/MicTranscriber.exe`
- Linux/macOS: `dist/MicTranscriber`

To produce a Windows `.exe`, run the build on Windows. This repository also
includes a GitHub Actions workflow, **Build Windows executable**, that can be
started manually from the Actions tab and uploads `MicTranscriber.exe` as an
artifact.

## Usage

1. Open the app.
2. Choose `Auto detect Polish or English`, `English`, or `Polish`.
3. Choose a Whisper model size.
4. Click **Start recording** and speak into your microphone.
5. Click **Stop and transcribe**.
6. Save or copy the transcript.

## Notes

- The app needs microphone permission from your operating system.
- Whisper models are downloaded from Hugging Face on first use, unless they are
  already cached on the machine.
- No OpenAI API key is required.
