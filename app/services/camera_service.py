from __future__ import annotations

"""
Camera service – captures images from an attached camera.

Supports:
  - Raspberry Pi Camera Module (via picamera2)
  - USB webcam (via OpenCV as fallback)
"""

import os
import time
from io import BytesIO
from typing import Generator

from loguru import logger

from app.config import settings


class CameraService:
    """Captures images from the attached camera."""

    def __init__(self) -> None:
        self._backend: str | None = None
        self._camera = None

    def initialise(self) -> None:
        """Detect and initialise the best available camera backend."""
        # Try picamera2 first (Raspberry Pi Camera Module)
        try:
            from picamera2 import Picamera2

            self._camera = Picamera2()
            self._camera.configure(
                self._camera.create_still_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
            )
            self._camera.start()
            time.sleep(1)  # warm-up
            self._backend = "picamera2"
            logger.info("Camera initialised (picamera2 – Pi Camera Module)")
            return
        except (ImportError, Exception) as exc:
            logger.debug(f"picamera2 not available: {exc}")

        # Fallback to OpenCV (USB webcam)
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

        logger.warning(
            "No camera backend available. "
            "Install picamera2 (Pi Camera) or opencv-python (USB webcam)."
        )

    @property
    def is_ready(self) -> bool:
        return self._camera is not None

    def capture(self) -> bytes:
        """
        Capture a single frame and return it as JPEG bytes.

        Raises RuntimeError if no camera is available.
        """
        if not self.is_ready:
            raise RuntimeError(
                "No camera available. Attach a camera and restart the app."
            )

        if self._backend == "picamera2":
            return self._capture_picamera2()
        elif self._backend == "opencv":
            return self._capture_opencv()
        else:
            raise RuntimeError(f"Unknown camera backend: {self._backend}")

    def _capture_picamera2(self) -> bytes:
        import numpy as np
        from PIL import Image

        arr = self._camera.capture_array()
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        logger.info("Image captured (picamera2)")
        return buf.getvalue()

    def _capture_opencv(self) -> bytes:
        import cv2

        ret, frame = self._camera.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam.")
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info("Image captured (OpenCV)")
        return buf.tobytes()

    def save_capture(self, image_bytes: bytes, directory: str = "data/captures") -> str:
        """Save captured image to disk and return the file path."""
        os.makedirs(directory, exist_ok=True)
        filename = f"capture_{int(time.time())}.jpg"
        filepath = os.path.join(directory, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        logger.debug(f"Image saved to '{filepath}'")
        return filepath

    def stream_frames(self, fps: int = 10) -> Generator[bytes, None, None]:
        """
        Yield MJPEG frames for live video streaming.

        Each yielded chunk is a complete multipart MJPEG frame boundary.
        """
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
                break

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

