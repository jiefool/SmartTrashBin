from __future__ import annotations

"""
Camera service – captures images from an attached camera.

Supports (in priority order):
  1. Raspberry Pi Camera Module (via picamera2)
  2. Raspberry Pi Camera via libcamera CLI (rpicam-still / libcamera-still)
  3. USB webcam (via OpenCV)
"""

import os
import shutil
import subprocess
import tempfile
import time
from io import BytesIO
from typing import Generator

from loguru import logger

from app.config import settings

# Temp file used by the libcamera CLI backend
_LIBCAMERA_TMP = os.path.join(tempfile.gettempdir(), "smartbin_capture.jpg")


class CameraService:
    """Captures images from the attached camera."""

    def __init__(self) -> None:
        self._backend: str | None = None
        self._camera = None
        self._libcamera_cmd: str | None = None

    def initialise(self) -> None:
        """Detect and initialise the best available camera backend."""

        # 1. Try picamera2 (Python library)
        try:
            from picamera2 import Picamera2

            self._camera = Picamera2()
            self._camera.configure(
                self._camera.create_still_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
            )
            self._camera.start()
            time.sleep(1)
            self._backend = "picamera2"
            logger.info("Camera initialised (picamera2 – Pi Camera Module)")
            return
        except (ImportError, Exception) as exc:
            logger.debug(f"picamera2 not available: {exc}")

        # 2. Try libcamera CLI (always available on Pi OS Bookworm)
        for cmd in ("rpicam-still", "libcamera-still"):
            if shutil.which(cmd):
                # Test that the camera actually works
                ret = subprocess.run(
                    [cmd, "-o", _LIBCAMERA_TMP, "-t", "1", "--nopreview", "-n"],
                    capture_output=True, timeout=10,
                )
                if ret.returncode == 0 and os.path.exists(_LIBCAMERA_TMP):
                    self._libcamera_cmd = cmd
                    self._camera = True  # sentinel so is_ready returns True
                    self._backend = "libcamera"
                    logger.info(f"Camera initialised (libcamera CLI – {cmd})")
                    return
                else:
                    logger.debug(f"{cmd} found but test capture failed: {ret.stderr.decode()}")

        # 3. Fallback to OpenCV (USB webcam)
        try:
            import cv2

            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self._camera = cap
                self._backend = "opencv"
                logger.info("Camera initialised (OpenCV – USB webcam)")
                return
            else:
                cap.release()
                logger.warning("OpenCV: no camera device found at index 0")
        except ImportError:
            logger.debug("OpenCV not available")

        logger.warning("No camera backend available.")

    @property
    def is_ready(self) -> bool:
        return self._camera is not None

    def capture(self) -> bytes:
        """Capture a single frame and return it as JPEG bytes."""
        if not self.is_ready:
            raise RuntimeError("No camera available. Attach a camera and restart the app.")

        if self._backend == "picamera2":
            return self._capture_picamera2()
        elif self._backend == "libcamera":
            return self._capture_libcamera()
        elif self._backend == "opencv":
            return self._capture_opencv()
        else:
            raise RuntimeError(f"Unknown camera backend: {self._backend}")

    def _capture_picamera2(self) -> bytes:
        from PIL import Image

        arr = self._camera.capture_array()
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _capture_libcamera(self) -> bytes:
        """Capture using rpicam-still or libcamera-still CLI."""
        subprocess.run(
            [self._libcamera_cmd, "-o", _LIBCAMERA_TMP,
             "-t", "1", "--nopreview", "-n",
             "--width", "640", "--height", "480", "--quality", "90"],
            capture_output=True, timeout=10,
        )
        if not os.path.exists(_LIBCAMERA_TMP):
            raise RuntimeError(f"{self._libcamera_cmd} failed to capture image.")
        with open(_LIBCAMERA_TMP, "rb") as f:
            return f.read()

    def _capture_opencv(self) -> bytes:
        import cv2

        for _ in range(3):  # retry up to 3 times
            ret, frame = self._camera.read()
            if ret:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                return buf.tobytes()
            time.sleep(0.1)
        raise RuntimeError("Failed to capture frame from webcam.")

    def save_capture(self, image_bytes: bytes, directory: str = "data/captures") -> str:
        """Save captured image to disk and return the file path."""
        os.makedirs(directory, exist_ok=True)
        filename = f"capture_{int(time.time())}.jpg"
        filepath = os.path.join(directory, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        logger.debug(f"Image saved to '{filepath}'")
        return filepath

    def stream_frames(self, fps: int = 5) -> Generator[bytes, None, None]:
        """Yield MJPEG frames for live video streaming."""
        delay = 1.0 / fps
        while self.is_ready:
            try:
                frame_bytes = self.capture()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
                time.sleep(delay)
            except Exception as exc:
                logger.error(f"Stream frame error: {exc}")
                time.sleep(1)  # wait before retrying instead of breaking

    def release(self) -> None:
        """Release camera resources."""
        if self._camera is not None:
            if self._backend == "picamera2":
                self._camera.stop()
            elif self._backend == "opencv":
                self._camera.release()
            self._camera = None
            logger.info("Camera released.")


# Singleton instance
camera = CameraService()

