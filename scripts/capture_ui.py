"""Capture offline-safe browser evidence from the running screening UI."""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
TARGETS = {
    "empty": IMAGES_DIR / "ui_empty.png",
    "light": IMAGES_DIR / "ui_result_light.png",
    "dark": IMAGES_DIR / "ui_result_dark.png",
    "error": IMAGES_DIR / "ui_error.png",
    "gif": IMAGES_DIR / "demo.gif",
}


def _hashes(directory: Path) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_model(port: int, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/api/v1/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                body = json.load(response)
            if body.get("model_loaded") is True:
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(1)
    raise TimeoutError("Timed out waiting for /api/v1/health model_loaded=true")


def _write_if_new(path: Path, data: bytes) -> None:
    if path.exists():
        print(f"Skipping existing UI capture: {path.name}")
        return
    path.write_bytes(data)


def _make_gif(path: Path, frames: list[Image.Image]) -> None:
    if path.exists():
        print(f"Skipping existing UI capture: {path.name}")
        return
    resized = []
    for frame in frames:
        image = frame.convert("RGB")
        image.thumbnail((960, 720), Image.Resampling.LANCZOS)
        resized.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    for scale in (1.0, 0.8, 0.65):
        images = resized if scale == 1.0 else [
            image.resize((int(image.width * scale), int(image.height * scale)))
            for image in resized
        ]
        output = BytesIO()
        images[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=1200,
            loop=0,
            optimize=True,
        )
        if output.tell() < 5 * 1024 * 1024:
            path.write_bytes(output.getvalue())
            return
    raise RuntimeError("Unable to reduce demo.gif below 5 MB")


def main() -> None:
    """Start the real app, capture the built-in example flow, and shut it down."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    before = _hashes(IMAGES_DIR)
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_model(port)
        frames: list[Image.Image] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, color_scheme="light")
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            empty = page.screenshot()
            frames.append(Image.open(BytesIO(empty)).copy())
            _write_if_new(TARGETS["empty"], empty)

            page.locator("#example").click()
            page.locator("#pasted .resume-box").nth(3).wait_for()
            page.locator("#rank").click()
            page.locator("#results .result").nth(0).wait_for(timeout=180_000)
            light = page.screenshot()
            frames.append(Image.open(BytesIO(light)).copy())
            _write_if_new(TARGETS["light"], light)

            page.locator("#theme").click()
            page.wait_for_timeout(300)
            dark = page.screenshot()
            frames.append(Image.open(BytesIO(dark)).copy())
            _write_if_new(TARGETS["dark"], dark)

            page.locator("#job").fill("too short")
            page.locator("#rank").click()
            page.wait_for_function(
                "() => document.querySelector('#error').textContent.length > 0"
            )
            error = page.screenshot()
            frames.append(Image.open(BytesIO(error)).copy())
            _write_if_new(TARGETS["error"], error)
            browser.close()
        _make_gif(TARGETS["gif"], frames)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        after = _hashes(IMAGES_DIR)
        for path, digest in before.items():
            if after.get(path) != digest:
                raise RuntimeError(f"Pre-existing image changed: {path.name}")
        print("Verified SHA-256 for all pre-existing images.")


if __name__ == "__main__":
    main()
