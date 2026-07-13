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
Data collection script - for building offline training datasets

Features:
1. Supports concurrent game screenshot collection from multiple devices
2. Standardized screenshot naming format for easy batch annotation
3. Only collects screenshots, does not call VLM (saves time and resources)
4. Automatically retries failed rounds
5. Records metadata for each game episode
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from adb_controller import ADBController


class DataCollector:
    """Game data collector"""

    def __init__(self, device_id: str, output_dir: str = "game_data_raw", debug: bool = False):
        self.device_id = device_id
        self.debug = debug

        # Create output directory (using safe filenames)
        safe_device_id = device_id.replace(":", "_").replace(".", "_")
        self.output_dir = Path(output_dir) / safe_device_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check existing episodes and auto-resume
        self.start_episode_id = self._find_next_episode_id()
        if self.start_episode_id > 1:
            print(f"[{device_id}] Existing data detected, continuing from Episode {self.start_episode_id}")

        # Initialize ADB controller
        print(f"[{device_id}] Connecting to Android device...")
        self.controller = ADBController(device_id=device_id)

        # Get screen resolution
        self.screen_width, self.screen_height = self.controller.get_screen_resolution()
        print(f"[{device_id}] Screen resolution: {self.screen_width}x{self.screen_height}")

        # Game metadata
        self.episodes = []

    def _find_next_episode_id(self) -> int:
        """
        Find the next available episode_id (to avoid overwriting existing data)

        Returns:
            Next episode_id (starting from 1)
        """
        existing_episodes = list(self.output_dir.glob("episode_*_metadata.json"))

        if not existing_episodes:
            return 1

        # Extract all existing episode_ids
        episode_ids = []
        for metadata_file in existing_episodes:
            # Filename format: episode_001_metadata.json
            match = re.match(r"episode_(\d+)_metadata\.json", metadata_file.name)
            if match:
                episode_ids.append(int(match.group(1)))

        if episode_ids:
            return max(episode_ids) + 1
        else:
            return 1

    def calculate_card_positions(self) -> List[Tuple[int, int]]:
        """Calculate click positions for the 3 option buttons"""
        # Based on screenshot analysis, option buttons are at approximately 65-68% screen height
        y = 860  # Adjusted coordinates (720x1280 screen, avoids triggering keyboard)
        positions = [
            (135, y),  # Left (option a)
            (360, y),  # Middle (option b)
            (585, y),  # Right (option c)
        ]
        return positions

    def calculate_next_button_position(self) -> Tuple[int, int]:
        """Calculate the position of the "next round" button"""
        # Next round button is at approximately 80-82% screen height
        return (360, 1040)

    def random_choice(self) -> int:
        """Randomly select an index (0, 1, 2)"""
        import random

        return random.choice([0, 1, 2])

    def capture_and_save(self, episode_id: int, round_num: int, suffix: str = "") -> Optional[str]:
        """
        Take a screenshot and save it using a standardized filename

        Filename format: episode_{ep}_round_{rd}_{suffix}.png
        Example: episode_001_round_03_question.png, episode_001_round_03_result.png

        Args:
            episode_id: episode number
            round_num: round number
            suffix: filename suffix, e.g. "question" or "result"

        Returns:
            Saved file path (relative path), or None on failure
        """
        try:
            screenshot = self.controller.capture_screenshot()

            # Standardized filename
            if suffix:
                filename = f"episode_{episode_id:03d}_round_{round_num:02d}_{suffix}.png"
            else:
                filename = f"episode_{episode_id:03d}_round_{round_num:02d}.png"
            filepath = self.output_dir / filename

            screenshot.save(filepath)

            if self.debug:
                print(f"[{self.device_id}] Screenshot saved: {filepath}")

            return str(filepath.relative_to(self.output_dir.parent))

        except Exception as e:
            print(f"⚠ [{self.device_id}] Screenshot failed: {e}")
            return None

    def check_card_color_changed(self) -> bool:
        """
        Simple check: wait briefly and retake screenshot to see if there is a color change.
        Uses a simplified approach: if no error occurs after the click, assume success.
        """
        time.sleep(0.8)
        return True  # Simplified handling, assume click always succeeds

    def play_one_round(self, episode_id: int, round_num: int) -> Optional[Dict]:
        """
        Play one round of the game and collect data

        Collects two screenshots:
        1. question.png - state before action (indicator light + number options)
        2. result.png - feedback after action (shows correct answer)

        Returns:
            Metadata dictionary for the round, or None on failure
        """
        print(f"[{self.device_id}] Round {round_num}/10")

        # Brief delay to avoid ADB conflicts during concurrent execution
        time.sleep(0.3)

        # 1. Screenshot 1: question (state before action)
        question_screenshot = self.capture_and_save(episode_id, round_num, "question")
        if question_screenshot is None:
            return None

        # 2. Randomly select a card to click
        selected_index = self.random_choice()

        if self.debug:
            print(f"[{self.device_id}] Randomly selected index: {selected_index}")

        # 3. Click the card
        positions = self.calculate_card_positions()
        x, y = positions[selected_index]

        max_retry = 3
        click_success = False

        for retry in range(max_retry):
            success = self.controller.tap(x, y, delay=1.0)
            if not success:
                if retry < max_retry - 1:
                    print(f"⚠ [{self.device_id}] Click failed, retrying {retry + 1}/{max_retry}")
                    time.sleep(0.5)
                    continue
                else:
                    print(f"⚠ [{self.device_id}] Click failed, skipping this round")
                    return None

            # Check if the click succeeded
            if self.check_card_color_changed():
                click_success = True
                break
            else:
                if retry < max_retry - 1:
                    print(f"⚠ [{self.device_id}] Click had no effect, retrying {retry + 1}/{max_retry}")
                    time.sleep(0.5)

        if not click_success:
            print(f"⚠ [{self.device_id}] Multiple clicks all failed")
            return None

        # 4. Wait for feedback to appear
        time.sleep(1.5)

        # 5. Screenshot 2: result (feedback after action, includes correct answer)
        # Add brief delay to avoid concurrent conflicts
        time.sleep(0.2)
        result_screenshot = self.capture_and_save(episode_id, round_num, "result")
        if result_screenshot is None:
            print(f"⚠ [{self.device_id}] Result screenshot failed")
            return None

        # 6. Click the "next round" button
        next_x, next_y = self.calculate_next_button_position()
        success = self.controller.tap(next_x, next_y, delay=1.0)

        if not success:
            print(f"⚠ [{self.device_id}] Click next round failed")
            return None

        # 7. Return metadata for this round
        metadata = {
            "round": round_num,
            "question_screenshot": question_screenshot,
            "result_screenshot": result_screenshot,
            "selected_index": selected_index,
            "click_position": [x, y],
            "timestamp": datetime.now().isoformat(),
        }

        return metadata

    def collect_one_episode(self, episode_id: int) -> Dict:
        """
        Collect data for one game episode (10 rounds)

        Returns:
            Metadata dictionary for this episode
        """
        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] Episode {episode_id} started")
        print(f"{'=' * 60}\n")

        episode_metadata = {
            "episode_id": episode_id,
            "device_id": self.device_id,
            "start_time": datetime.now().isoformat(),
            "rounds": [],
            "completed_rounds": 0,
            "success": False,
        }

        # Collect 10 rounds of data
        completed_rounds = 0
        attempt_count = 0
        max_attempts = 20  # Maximum 20 attempts

        while completed_rounds < 10 and attempt_count < max_attempts:
            attempt_count += 1
            round_num = completed_rounds + 1

            round_metadata = self.play_one_round(episode_id, round_num)

            if round_metadata is not None:
                episode_metadata["rounds"].append(round_metadata)
                completed_rounds += 1
                print(f"✓ [{self.device_id}] Round {completed_rounds}/10 completed")

                # Wait between rounds
                if completed_rounds < 10:
                    time.sleep(1.0)
            else:
                print(f"⚠ [{self.device_id}] Round {round_num} failed, retrying...")
                time.sleep(1.5)

        episode_metadata["completed_rounds"] = completed_rounds
        episode_metadata["success"] = completed_rounds == 10
        episode_metadata["end_time"] = datetime.now().isoformat()

        # Capture final score screen
        time.sleep(2.0)
        final_screenshot_path = self.capture_and_save(episode_id, 99, "final")  # Use 99 to denote final
        if final_screenshot_path:
            episode_metadata["final_screenshot"] = final_screenshot_path

        # Save metadata for this episode
        metadata_file = self.output_dir / f"episode_{episode_id:03d}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(episode_metadata, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] Episode {episode_id} completed")
        print(f"[{self.device_id}] Successful rounds: {completed_rounds}/10")
        print(f"[{self.device_id}] Metadata saved: {metadata_file}")
        print(f"{'=' * 60}\n")

        self.episodes.append(episode_metadata)
        return episode_metadata

    def refresh_browser(self):
        """Refresh the browser page to prepare for the next episode"""
        print(f"[{self.device_id}] Refreshing browser...")

        # Click the refresh button
        refresh_button_x = 380
        refresh_button_y = 130
        self.controller.tap(refresh_button_x, refresh_button_y, delay=1.0)

        time.sleep(3.0)  # Wait for page to load
        print(f"[{self.device_id}] Browser refreshed")

    def collect_data(self, num_episodes: int) -> List[Dict]:
        """
        Collect data for multiple game episodes

        Args:
            num_episodes: Number of episodes to collect

        Returns:
            List of metadata dictionaries for all episodes
        """
        # Start from the resumed episode ID
        for i in range(num_episodes):
            episode_id = self.start_episode_id + i
            self.collect_one_episode(episode_id)

            # Refresh browser between episodes (not needed after the last one)
            if i < num_episodes - 1:
                self.refresh_browser()
                time.sleep(2.0)

        # Save summary information
        summary = {
            "device_id": self.device_id,
            "total_episodes": num_episodes,
            "successful_episodes": sum(1 for ep in self.episodes if ep["success"]),
            "total_rounds_collected": sum(ep["completed_rounds"] for ep in self.episodes),
            "collection_time": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "episodes": self.episodes,
        }

        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f"[{self.device_id}] Data collection complete!")
        print(f"[{self.device_id}] Total episodes: {num_episodes}")
        print(f"[{self.device_id}] Successful episodes: {summary['successful_episodes']}")
        print(f"[{self.device_id}] Total rounds: {summary['total_rounds_collected']}")
        print(f"[{self.device_id}] Summary file: {summary_file}")
        print(f"{'=' * 60}\n")

        return self.episodes


def collect_from_device(device_id: str, num_episodes: int, output_dir: str, debug: bool) -> Dict:
    """
    Collect data from a single device (for concurrent execution)

    Returns:
        Collection summary information
    """
    try:
        collector = DataCollector(device_id=device_id, output_dir=output_dir, debug=debug)

        collector.collect_data(num_episodes)

        # Read the summary file
        summary_file = collector.output_dir / "collection_summary.json"
        with open(summary_file, encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"⚠ Device {device_id} collection failed: {e}")
        import traceback

        traceback.print_exc()
        return {"device_id": device_id, "error": str(e), "success": False}


def main():
    parser = argparse.ArgumentParser(description="Game data collection script (for offline training)")

    # Device configuration
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        required=True,
        help="Android device address list, e.g.: 101.43.137.83:5555 192.168.1.100:5555",
    )

    # Collection configuration
    parser.add_argument("--episodes", type=int, default=10, help="Number of game episodes to collect per device (default 10)")

    parser.add_argument("--output-dir", type=str, default="game_data_raw", help="Output directory (default game_data_raw)")

    # Execution mode
    parser.add_argument("--parallel", action="store_true", help="Run multiple devices concurrently (default is sequential)")

    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of threads for concurrent execution (default 4)")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("=" * 60)
    print("Game Data Collection Script")
    print("=" * 60)
    print(f"Number of devices: {len(args.devices)}")
    print(f"Episodes per device: {args.episodes}")
    print(f"Estimated total rounds: {len(args.devices) * args.episodes * 10}")
    print(f"Output directory: {args.output_dir}")
    print(f"Execution mode: {'concurrent' if args.parallel else 'sequential'}")
    print("=" * 60)
    print()

    start_time = time.time()
    all_summaries = []

    if args.parallel and len(args.devices) > 1:
        # Concurrent execution
        print(f"Using {min(args.max_workers, len(args.devices))} threads to collect data concurrently...\n")

        with ThreadPoolExecutor(max_workers=min(args.max_workers, len(args.devices))) as executor:
            # Submit all tasks
            future_to_device = {
                executor.submit(collect_from_device, device_id, args.episodes, args.output_dir, args.debug): device_id
                for device_id in args.devices
            }

            # Wait for completion
            for future in as_completed(future_to_device):
                device_id = future_to_device[future]
                try:
                    summary = future.result()
                    all_summaries.append(summary)
                    print(f"✓ Device {device_id} data collection complete")
                except Exception as e:
                    print(f"⚠ Device {device_id} encountered an exception: {e}")
    else:
        # Sequential execution
        for device_id in args.devices:
            print(f"\nProcessing device: {device_id}")
            print("-" * 60)

            summary = collect_from_device(device_id, args.episodes, args.output_dir, args.debug)
            all_summaries.append(summary)

    # Generate overall summary
    elapsed_time = time.time() - start_time
    total_summary = {
        "total_devices": len(args.devices),
        "episodes_per_device": args.episodes,
        "total_episodes_collected": sum(s.get("total_episodes", 0) for s in all_summaries),
        "successful_episodes": sum(s.get("successful_episodes", 0) for s in all_summaries),
        "total_rounds_collected": sum(s.get("total_rounds_collected", 0) for s in all_summaries),
        "collection_time_seconds": elapsed_time,
        "output_dir": args.output_dir,
        "timestamp": datetime.now().isoformat(),
        "device_summaries": all_summaries,
    }

    # Save overall summary
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_summary_file = output_path / "total_summary.json"
    with open(total_summary_file, "w", encoding="utf-8") as f:
        json.dump(total_summary, f, ensure_ascii=False, indent=2)

    # Print final results
    print("\n" + "=" * 60)
    print("All data collection complete!")
    print("=" * 60)
    print(f"Total devices: {total_summary['total_devices']}")
    print(f"Total episodes: {total_summary['total_episodes_collected']}")
    print(f"Successful episodes: {total_summary['successful_episodes']}")
    print(f"Total rounds: {total_summary['total_rounds_collected']}")
    print(f"Elapsed time: {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)")
    print(f"Output directory: {args.output_dir}")
    print(f"Total summary file: {total_summary_file}")
    print("=" * 60)
    print("\nNext step: Use the annotation script to batch-annotate the collected screenshots")


if __name__ == "__main__":
    main()
