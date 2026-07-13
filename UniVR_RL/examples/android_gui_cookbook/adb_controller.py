# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ADB Device Controller

Features:
- Connect to Android devices
- Execute tap, swipe, input, and other operations
- Capture screenshots
- Detect device status
"""

import io
import subprocess
import time
from typing import Optional, Tuple

from PIL import Image


class ADBController:
    """Android Debug Bridge controller."""

    def __init__(self, device_id: str = "emulator-5554"):
        """
        Initialize the ADB controller.

        Args:
            device_id: Device ID (visible via `adb devices`)
        """
        self.device_id = device_id
        self._check_connection()

    def _check_connection(self):
        """Check device connection."""
        try:
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)

            if self.device_id not in result.stdout:
                raise ConnectionError(f"Device {self.device_id} is not connected. Run 'adb devices' to see available devices.")

            print(f"✓ Device {self.device_id} connected")

        except FileNotFoundError:
            raise RuntimeError("ADB is not installed or not in PATH. Please install Android SDK Platform-Tools.")
        except subprocess.TimeoutExpired:
            raise TimeoutError("ADB connection timed out. Please check device status.")

    def execute_command(self, command: str, timeout: int = 10) -> str:
        """
        Execute an ADB command.

        Args:
            command: ADB command (without the 'adb -s device_id' prefix)
            timeout: Timeout in seconds

        Returns:
            Command output string
        """
        full_command = f"adb -s {self.device_id} {command}"

        try:
            result = subprocess.run(full_command.split(), capture_output=True, text=True, timeout=timeout)

            if result.returncode != 0:
                raise RuntimeError(f"Command failed: {result.stderr}")

            return result.stdout

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out: {command}")

    def capture_screenshot(self, save_path: Optional[str] = None) -> Image.Image:
        """
        Capture a screenshot.

        Args:
            save_path: Optional path to save the screenshot

        Returns:
            PIL Image object
        """
        try:
            # Use screencap command
            cmd = f"adb -s {self.device_id} exec-out screencap -p"
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                timeout=15,  # increased timeout for remote devices or concurrent use
            )

            if result.returncode != 0:
                raise RuntimeError("Screenshot failed")

            # Convert byte stream to PIL Image
            image = Image.open(io.BytesIO(result.stdout))

            if save_path:
                image.save(save_path)
                print(f"✓ Screenshot saved: {save_path}")

            return image

        except Exception as e:
            raise RuntimeError(f"Screenshot failed: {e}")

    def tap(self, x: int, y: int, delay: float = 0.5) -> bool:
        """
        Tap screen coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            delay: Wait time after tap (seconds)

        Returns:
            True if successful
        """
        try:
            self.execute_command(f"shell input tap {x} {y}")
            time.sleep(delay)
            print(f"✓ Tapped: ({x}, {y})")
            return True

        except Exception as e:
            print(f"✗ Tap failed: {e}")
            return False

    def get_screen_resolution(self) -> Tuple[int, int]:
        """
        Get screen resolution.

        Returns:
            (width, height)
        """
        try:
            output = self.execute_command("shell wm size")
            # Output format: Physical size: 1080x2400
            size_str = output.split(":")[-1].strip()
            width, height = map(int, size_str.split("x"))
            return width, height

        except Exception as e:
            print(f"⚠ Cannot get resolution, using default (1080, 2400): {e}")
            return 1080, 2400
