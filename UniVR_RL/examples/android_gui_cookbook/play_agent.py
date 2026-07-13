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
Agent for automatically playing the number selection game

Features:
1. Supports both Ollama and vLLM model services
2. Supports concurrent/sequential execution across multiple Android devices
3. Automatically detects game state, indicator lights, and numbers
4. Automatically makes decisions and executes actions
5. Records game results and screenshots
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from adb_controller import ADBController
from PIL import Image
from vlm_client import VLMClient


class GameAgent:
    """Number selection game Agent"""

    def __init__(
        self, device_id: str, vlm_client: VLMClient, screenshot_dir: str = "game_screenshots", debug: bool = False
    ):
        self.device_id = device_id
        self.vlm_client = vlm_client
        self.debug = debug

        # Create screenshot directory
        self.screenshot_dir = Path(screenshot_dir) / device_id.replace(":", "_")
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ADB controller
        print(f"[{device_id}] Connecting to Android device...")
        self.controller = ADBController(device_id=device_id)

        # Get screen resolution
        self.screen_width, self.screen_height = self.controller.get_screen_resolution()
        print(f"[{device_id}] Screen resolution: {self.screen_width}x{self.screen_height}")

        # Game state
        self.current_round = 0
        self.game_results = []

    def calculate_card_positions(self) -> List[Tuple[int, int]]:
        """Calculate click positions for the 3 cards"""
        # Based on actual testing, precise coordinates for 720x1280 screen
        y = 905  # Empirically measured coordinates

        positions = [
            (135, y),  # Left number
            (360, y),  # Middle number
            (585, y),  # Right number
        ]
        return positions

    def calculate_next_button_position(self) -> Tuple[int, int]:
        """Calculate the position of the "next round" button"""
        # Next round button coordinates (empirically measured)
        x = 360
        y = 1070
        return (x, y)

    def recognize_score(self, screenshot: Image.Image) -> Optional[int]:
        """
        Use VLM to recognize the game score in a screenshot

        Args:
            screenshot: game screenshot

        Returns:
            Recognized score value, or None if recognition fails
        """
        prompt = """Please carefully observe this game screenshot and identify the large number (current score) in the center of the pink/red gradient card.

Only reply with the number, no other explanation needed."""

        try:
            response = self.vlm_client.query(screenshot, prompt)

            if self.debug:
                print(f"[{self.device_id}] Score recognition VLM output: {response}")

            # Extract number from response
            # Try multiple pattern matches
            patterns = [
                r"(\d+)",  # Any number
                r"score[is:：]\s*(\d+)",
                r"number[is:：]\s*(\d+)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, response)
                if matches:
                    score = int(matches[0])
                    print(f"[{self.device_id}] Detected score: {score}")
                    return score

            print(f"⚠ [{self.device_id}] Unable to extract score from VLM output")
            return None

        except Exception as e:
            print(f"⚠ [{self.device_id}] Score recognition failed: {e}")
            return None

    def capture_screenshot(self, round_num: int) -> Image.Image:
        """Capture the screen and save it"""
        screenshot = self.controller.capture_screenshot()

        # Save screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"round_{round_num:02d}_{timestamp}.png"
        filepath = self.screenshot_dir / filename
        screenshot.save(filepath)

        if self.debug:
            print(f"[{self.device_id}] Screenshot saved: {filepath}")

        return screenshot

    def check_click_success(self, screenshot: Image.Image) -> bool:
        """
        Check whether the click succeeded (whether a card's color changed)

        Args:
            screenshot: screenshot taken after the click

        Returns:
            True if click succeeded, False otherwise
        """
        prompt = """Please observe this game screenshot and determine whether any of the three number cards has changed color (no longer the original purple/blue gradient, but changed to red, green, or yellow).

If a card's color has changed, answer "yes"; if all card colors remain unchanged, answer "no"."""

        try:
            response = self.vlm_client.query(screenshot, prompt)

            if self.debug:
                print(f"[{self.device_id}] Click success check VLM output: {response}")

            # Determine whether the click succeeded
            if "yes" in response.lower() or "changed" in response.lower() or "success" in response.lower():
                return True
            else:
                return False

        except Exception as e:
            print(f"⚠ [{self.device_id}] Click success check failed: {e}")
            return False

    def parse_vlm_response(self, response: str) -> Optional[int]:
        """
        Parse VLM output and extract the selected index

        Returns:
            Selected index (0-2), or None if parsing fails
        """
        # Strategy 1: Extract <action>select(N)</action>
        pattern = r"<action>select\((\d+)\)</action>"
        matches = re.findall(pattern, response, re.IGNORECASE)

        if matches:
            index = int(matches[0])
            if 0 <= index <= 2:
                return index

        # Strategy 2: Extract "selected index: N" or similar text
        index_patterns = [
            r"selected?\s*index[:：]\s*(\d+)",
            r"index[:：]\s*(\d+)",
            r"select\((\d+)\)",
            r"select\s*(\d+)",
            r"number\s*(\d+)",
            r"answer[:：]\s*(\d+)",
            r"option\s*([abc])",
        ]

        for pattern in index_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                match_str = matches[0]
                # Handle option a/b/c
                if match_str.lower() in ["a", "b", "c"]:
                    index = ord(match_str.lower()) - ord("a")
                else:
                    index = int(match_str)

                if 0 <= index <= 2:
                    return index

        print(f"⚠ [{self.device_id}] VLM output parsing failed, unable to extract index")
        return None

    def make_decision(self, screenshot: Image.Image) -> Optional[int]:
        """
        Make a decision based on the screenshot

        Returns:
            Selected index (0-2), or None on failure
        """
        prompt = """This is a screenshot of a number selection game.

Game rules:
- There are 3 indicator lights (circular) at the top of the screen
- Green light on: select the largest number (+10 points)
- Red light on: select the smallest number (+10 points)
- Yellow light on: select the middle number (+10 points)
- There are 3 number cards in the center of the screen (arranged left to right), labeled "option a", "option b", "option c"

Task:
1. Identify which indicator light is on (green/red/yellow)
2. Identify the numbers on the 3 number cards (option a, option b, option c)
3. Select the correct number based on the lit light rule

Please answer in the following format:
1. Lit light color: [green/red/yellow]
2. Recognized numbers: [number of option a, number of option b, number of option c]
3. Should select: [largest/smallest/middle]
4. Selected index: N (0=option a/left, 1=option b/middle, 2=option c/right)

Finally output your choice in the following format:
<action>select(N)</action>

Where N is 0, 1, or 2.
"""

        print(f"[{self.device_id}] VLM inference in progress...")
        response = self.vlm_client.query(screenshot, prompt)

        if self.debug:
            print("\n--- VLM output ---")
            print(response)
            print("--- Output end ---\n")

        # Parse the response
        selected_index = self.parse_vlm_response(response)

        if selected_index is not None:
            print(f"[{self.device_id}] VLM decision: selected index {selected_index}")

        return selected_index

    def play_one_round(self, round_num: int) -> bool:
        """
        Play one round of the game

        Returns:
            Whether the round was completed successfully
        """
        print(f"\n[{self.device_id}] ========== Round {round_num}/10 ==========")

        # 1. Take screenshot
        print(f"[{self.device_id}] [1/5] Taking screenshot...")
        screenshot = self.capture_screenshot(round_num)

        # 2. VLM decision
        print(f"[{self.device_id}] [2/5] VLM decision...")
        selected_index = self.make_decision(screenshot)

        if selected_index is None:
            print(f"⚠ [{self.device_id}] Decision failed, skipping this round")
            return False

        # 3. Execute action: click the card and verify success
        print(f"[{self.device_id}] [3/5] Clicking card...")
        positions = self.calculate_card_positions()
        x, y = positions[selected_index]

        max_retry = 3
        click_success = False

        for retry in range(max_retry):
            # Click the card
            success = self.controller.tap(x, y, delay=1.5)
            if not success:
                print(f"⚠ [{self.device_id}] Card click failed")
                return False

            print(f"[{self.device_id}] Clicked position: ({x}, {y})")

            # Wait briefly for the interface to react
            time.sleep(1.0)

            # Take screenshot to check if click succeeded, and save it
            check_screenshot = self.controller.capture_screenshot()

            # Save screenshot after click
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"round_{round_num:02d}_after_click_{timestamp}.png"
            filepath = self.screenshot_dir / filename
            check_screenshot.save(filepath)
            if self.debug:
                print(f"[{self.device_id}] Post-click screenshot saved: {filepath}")

            click_success = self.check_click_success(check_screenshot)

            if click_success:
                print(f"[{self.device_id}] ✓ Click successful, card color has changed")
                break
            else:
                print(f"⚠ [{self.device_id}] Click unsuccessful (attempt {retry + 1}), retrying...")
                time.sleep(0.5)

        if not click_success:
            print(f"⚠ [{self.device_id}] Multiple clicks all failed, skipping this round")
            return False

        # 4. Wait for feedback to appear
        print(f"[{self.device_id}] [4/5] Waiting for feedback...")
        time.sleep(1.5)

        # 5. Click the "next round" button
        print(f"[{self.device_id}] [5/5] Clicking next round...")
        next_x, next_y = self.calculate_next_button_position()
        success = self.controller.tap(next_x, next_y, delay=1.5)

        if not success:
            print(f"⚠ [{self.device_id}] Click next round button failed")
            return False

        print(f"[{self.device_id}] Clicked next round button: ({next_x}, {next_y})")

        return True

    def capture_final_score(self) -> Image.Image:
        """Capture the final score screenshot at game end"""
        print(f"[{self.device_id}] Capturing final score...")
        time.sleep(2.0)  # Wait for game end animation

        screenshot = self.controller.capture_screenshot()

        # Save the final screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_score_{timestamp}.png"
        filepath = self.screenshot_dir / filename
        screenshot.save(filepath)

        print(f"[{self.device_id}] Final score screenshot saved: {filepath}")

        return screenshot

    def refresh_browser(self):
        """Refresh the browser to prepare for the next game episode"""
        print(f"[{self.device_id}] Refreshing browser...")

        # Method 1: Press back key to close any keyboard that may have appeared
        os.system(f"adb -s {self.device_id} shell input keyevent 4")  # KEYCODE_BACK
        time.sleep(0.5)

        # Method 2: Click the refresh button in the upper right corner (next to the three-dot button)
        # Based on screenshot, refresh button is to the right of the address bar, around x=540
        # For a 720x1280 screen, the refresh button is approximately at (540, 140)
        refresh_button_x = 380  # Forward button position to the right of the address bar
        refresh_button_y = 130

        self.controller.tap(refresh_button_x, refresh_button_y, delay=1.0)

        time.sleep(3.0)  # Wait for page to load
        print(f"[{self.device_id}] Browser refreshed")

    def save_results(self, final_screenshot: Image.Image, final_score: Optional[int] = None):
        """Save game results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result = {
            "device_id": self.device_id,
            "timestamp": timestamp,
            "total_rounds": 10,
            "final_score": final_score,
            "screenshot_dir": str(self.screenshot_dir),
            "final_screenshot": str(self.screenshot_dir / f"final_score_{timestamp}.png"),
            "model_type": self.vlm_client.model_type,
            "model_name": self.vlm_client.model_name,
        }

        # Save results
        result_file = self.screenshot_dir / f"result_{timestamp}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[{self.device_id}] Game results saved: {result_file}")
        if final_score is not None:
            print(f"[{self.device_id}] Final score: {final_score}")

        self.game_results.append(result)

    def run_game(self) -> Dict:
        """
        Run a complete game (10 rounds)

        Returns:
            Game result dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] Starting game")
        print(f"{'=' * 60}\n")

        # Play 10 rounds, ensuring each round succeeds
        completed_rounds = 0
        max_total_attempts = 20  # Maximum 20 attempts to avoid infinite loops
        total_attempts = 0

        while completed_rounds < 10 and total_attempts < max_total_attempts:
            total_attempts += 1

            print(
                f"\n[{self.device_id}] ========== Attempting round {completed_rounds + 1} (total attempts: {total_attempts}) =========="
            )

            success = self.play_one_round(completed_rounds + 1)

            if success:
                completed_rounds += 1
                print(f"✓ [{self.device_id}] Round {completed_rounds} complete!")
            else:
                print(f"⚠ [{self.device_id}] Round execution failed, will retry...")
                time.sleep(2.0)  # Wait a bit longer after failure
                continue

            # Wait between rounds
            if completed_rounds < 10:
                time.sleep(1.5)

        if completed_rounds < 10:
            print(f"\n⚠ [{self.device_id}] Warning: failed to complete 10 rounds, only completed {completed_rounds} rounds")

        # Game over, capture final score
        final_screenshot = self.capture_final_score()

        # Use VLM to recognize the final score
        print(f"[{self.device_id}] Recognizing final score...")
        final_score = self.recognize_score(final_screenshot)

        if final_score is not None:
            print(f"\n{'=' * 60}")
            print(f"[{self.device_id}] Final score: {final_score}")
            print(f"{'=' * 60}\n")
        else:
            print(f"⚠ [{self.device_id}] Failed to recognize final score")

        # Save results
        self.save_results(final_screenshot, final_score)

        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] Game complete!")
        print(f"[{self.device_id}] Completed rounds: {completed_rounds}/10")
        print(f"[{self.device_id}] Please manually refresh the browser to start the next episode")
        print(f"{'=' * 60}\n")

        return self.game_results[-1] if self.game_results else {}


def main():
    parser = argparse.ArgumentParser(description="Agent for automatically playing the number selection game")

    # Model configuration
    parser.add_argument(
        "--model-type", type=str, choices=["ollama", "vllm"], default="ollama", help="Model service type: ollama or vllm"
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:11434",
        help="Model API address (Ollama: http://localhost:11434, vLLM: http://localhost:8000)",
    )

    parser.add_argument("--model-name", type=str, default="qwen2.5vl:3b", help="Model name")

    # Android device configuration
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["101.43.137.83:5555"],
        help="Android device address list, e.g.: 101.43.137.83:5555 192.168.1.100:5555",
    )

    # Other configuration
    parser.add_argument("--screenshot-dir", type=str, default="game_screenshots", help="Screenshot save directory")

    parser.add_argument("--episodes", type=int, default=1, help="Number of game episodes to run per device")

    parser.add_argument("--parallel", action="store_true", help="Process multiple devices concurrently (default is sequential)")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Initialize VLM client
    print("=" * 60)
    print("Initializing VLM client")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"API address: {args.api_url}")
    print(f"Model name: {args.model_name}")
    print()

    vlm_client = VLMClient(model_type=args.model_type, api_url=args.api_url, model_name=args.model_name)

    # Process multiple devices
    all_results = []

    if args.parallel:
        # TODO: Implement concurrent processing (using threading or multiprocessing)
        print("⚠ Concurrent mode not yet implemented, using sequential mode")
        args.parallel = False

    # Sequential processing
    for device_id in args.devices:
        print(f"\n{'=' * 60}")
        print(f"Processing device: {device_id}")
        print(f"{'=' * 60}\n")

        # Create Agent
        agent = GameAgent(
            device_id=device_id, vlm_client=vlm_client, screenshot_dir=args.screenshot_dir, debug=args.debug
        )

        # Run multiple game episodes
        for episode in range(1, args.episodes + 1):
            if args.episodes > 1:
                print(f"\n--- Episode {episode}/{args.episodes} ---\n")

            result = agent.run_game()
            all_results.append(result)

            # Wait between episodes
            if episode < args.episodes:
                print("\nWaiting 5 seconds before starting next episode...\n")
                time.sleep(5)

    # Print summary
    print("\n" + "=" * 60)
    print("All games complete!")
    print("=" * 60)
    print(f"Total completed: {len(all_results)} game episodes")
    print(f"Devices involved: {len(args.devices)}")
    print("\nResult files have been saved to their respective screenshot directories")


if __name__ == "__main__":
    main()
