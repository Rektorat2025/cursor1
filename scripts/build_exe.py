from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            str(repo_root / "MicTranscriber.spec"),
        ],
        cwd=repo_root,
        check=True,
    )


if __name__ == "__main__":
    main()
