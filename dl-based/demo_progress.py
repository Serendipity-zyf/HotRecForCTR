"""
Demo script to showcase the progress bar functionality.
"""

import time
import random

from utils.logger import ColorLogger
from utils.progress import ProgressBar


def slow_process(item: int) -> int:
    """A slow process to demonstrate progress bars."""
    time.sleep(random.uniform(0.1, 0.3))
    return item * 2


def demo_progress():
    """Demonstrate the progress bar functionality."""
    logger = ColorLogger(name="ProgressDemo")
    logger.info("Starting progress bar demo")

    print("\n" + "=" * 50)
    print("PROGRESS BAR DEMO")
    print("=" * 50)

    # Basic progress bar using context manager
    print("\nBasic progress bar using context manager:")
    items = list(range(20))
    with ProgressBar(total=len(items), title="Processing items") as bar:
        for _ in items:
            time.sleep(0.1)
            bar()

    # Process items with progress
    print("\nProcessing items with progress:")
    progress_bar = ProgressBar(total=15, title="Doubling numbers")
    results = progress_bar.process_items(range(15), slow_process)
    print(f"Results: {results[:5]}... (total: {len(results)} items)")

    # Timed progress
    print("\nTimed progress bar:")
    ProgressBar.timed(seconds=3, title="Waiting", step=0.1)

    # Multiple progress bars
    print("\nMultiple progress bars:")
    tasks = {"Task 1": 10, "Task 2": 15, "Task 3": 5}

    bars = ProgressBar.multi_task(tasks)

    try:
        # Simulate progress for each task
        for _ in range(tasks["Task 1"]):
            time.sleep(0.1)
            bars["Task 1"]()

        for _ in range(tasks["Task 2"]):
            time.sleep(0.07)
            bars["Task 2"]()

        for _ in range(tasks["Task 3"]):
            time.sleep(0.2)
            bars["Task 3"]()
    finally:
        # Clean up the bars
        bars["_cleanup"]()

    # Show available styles
    print("\nSome available spinner styles:")
    spinners = ProgressBar.get_available_spinners()
    print(", ".join(spinners[:10]) + "...")

    print("\nSome available bar styles:")
    bars = ProgressBar.get_available_bars()
    print(", ".join(bars[:10]) + "...")

    # Custom styling
    print("\nCustom styling:")
    with ProgressBar(
        total=10,
        title="Custom style",
        spinner="dots_waves",
        bar="smooth",
        title_color="\033[95m",  # Purple
    ) as bar:
        for i in range(10):
            time.sleep(0.1)
            bar()

    # Manual update with steps
    print("\nManual update with steps:")
    progress = ProgressBar(total=100, title="Manual steps")
    with progress:
        for _ in range(10):
            time.sleep(0.1)
            # Update by 10 steps at once
            progress.update(10)

    print("\nDemo completed!")


if __name__ == "__main__":
    demo_progress()
