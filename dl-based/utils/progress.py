"""
Enhanced progress bar utilities using alive_progress.
"""

import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import TypeVar

from alive_progress import alive_bar
from alive_progress import config_handler
from alive_progress.styles import BARS
from alive_progress.styles import SPINNERS
from colorama import Fore
from colorama import Style

# Configure default styles for alive_progress
config_handler.set_global(
    spinner="waves",  # A nice spinner
    bar="classic",  # Classic bar style
    unknown="brackets",  # Style for unknown total
    theme="smooth",  # Smooth theme
    stats_end=True,  # Show stats at the end
)

# Type variable for generic return type
T = TypeVar("T")


class ProgressBar(object):
    """
    A class that provides enhanced progress bar functionality using alive_progress.

    This class encapsulates various progress bar utilities including:
    - Basic progress bars
    - Processing items with progress tracking
    - Timed progress bars
    - Multi-task progress tracking

    Attributes:
        default_spinner (str): The default spinner style to use.
        default_bar (str): The default bar style to use.
        default_theme (str): The default theme to use.
        default_title_color (str): The default color for the title.
    """

    default_spinner = "waves"
    default_bar = "classic"
    default_theme = "smooth"
    default_title_color = Fore.CYAN

    @classmethod
    def get_available_spinners(cls) -> List[str]:
        """
        Get a list of all available spinner styles.

        Returns:
            A list of spinner style names.
        """
        return list(SPINNERS.keys())

    @classmethod
    def get_available_bars(cls) -> List[str]:
        """
        Get a list of all available bar styles.

        Returns:
            A list of bar style names.
        """
        return list(BARS.keys())

    def __init__(
        self,
        total: Optional[int] = None,
        title: Optional[str] = None,
        spinner: str = None,
        bar: str = None,
        theme: str = None,
        enrich_print: bool = True,
        manual: bool = False,
        stats: bool = True,
        title_color: str = None,
        **kwargs,
    ):
        """
        Initialize a ProgressBar instance.

        Args:
            total: The total number of items to process.
            title: The title of the progress bar.
            spinner: The spinner style to use.
            bar: The bar style to use.
            theme: The theme to use.
            enrich_print: Whether to enrich print statements.
            manual: Whether to use manual mode.
            stats: Whether to show stats.
            title_color: The color for the title.
            **kwargs: Additional arguments to pass to alive_bar.
        """
        self.total = total
        self.title = title
        self.spinner = spinner or self.default_spinner
        self.bar = bar or self.default_bar
        self.theme = theme or self.default_theme
        self.enrich_print = enrich_print
        self.manual = manual
        self.stats = stats
        self.title_color = title_color or self.default_title_color
        self.kwargs = kwargs
        self.bar_context = None
        self.bar_func = None

    def __enter__(self):
        """
        Enter the context manager and create the progress bar.

        Returns:
            The progress bar update function.
        """
        # Apply colors to title if provided
        colored_title = f"{self.title_color}{self.title}{Style.RESET_ALL}" if self.title else None

        # Create the progress bar
        self.bar_context = alive_bar(
            total=self.total,
            title=colored_title,
            spinner=self.spinner,
            bar=self.bar,
            theme=self.theme,
            enrich_print=self.enrich_print,
            manual=self.manual,
            stats=self.stats,
            **self.kwargs,
        )

        # Enter the context and get the bar function
        self.bar_func = self.bar_context.__enter__()

        return self.bar_func

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and clean up the progress bar.

        Args:
            exc_type: The exception type, if any.
            exc_val: The exception value, if any.
            exc_tb: The exception traceback, if any.
        """
        if self.bar_context:
            self.bar_context.__exit__(exc_type, exc_val, exc_tb)

    def update(self, steps: int = 1):
        """
        Update the progress bar by the specified number of steps.

        Args:
            steps: The number of steps to advance the progress bar.
        """
        if self.bar_func:
            for _ in range(steps):
                self.bar_func()

    def process_items(self, items: Iterable[T], process_func: Callable[[T], Any]) -> List[Any]:
        """
        Process a collection of items with a progress bar.

        Args:
            items: The items to process.
            process_func: The function to apply to each item.

        Returns:
            A list of processed items.
        """
        results = []

        with self:
            for item in items:
                result = process_func(item)
                results.append(result)
                self.update()

        return results

    @classmethod
    def timed(cls, seconds: int, title: str = "Processing", step: float = 0.1, **kwargs):
        """
        Create a progress bar that runs for a specified number of seconds.

        Args:
            seconds: The number of seconds to run.
            title: The title for the progress bar.
            step: The time step in seconds.
            **kwargs: Additional arguments to pass to the ProgressBar constructor.

        Returns:
            A ProgressBar instance.
        """
        steps = int(seconds / step)

        progress_bar = cls(total=steps, title=title, **kwargs)

        with progress_bar:
            for _ in range(steps):
                time.sleep(step)
                progress_bar.update()

        return progress_bar

    @classmethod
    def multi_task(cls, tasks: Dict[str, int], **kwargs) -> Dict[str, Callable]:
        """
        Create a simulated multi-progress bar system.

        Note: alive_progress doesn't support nested progress bars, so this method
        provides a simpler alternative that uses a single progress bar with a title
        that changes based on the current task.

        Args:
            tasks: A dictionary mapping task names to their total steps.
            **kwargs: Additional arguments to pass to the ProgressBar constructor.

        Returns:
            A dictionary mapping task names to their progress bar update functions.
        """
        # Calculate total steps across all tasks
        total_steps = sum(tasks.values())

        # Create a single progress bar
        progress_bar = cls(total=total_steps, title="Multiple tasks", **kwargs)
        main_bar = progress_bar.__enter__()

        # Track progress for each task
        task_progress = {name: 0 for name in tasks}

        # Create update functions for each task
        result = {}
        for task_name in tasks:

            def make_updater(name):
                def update():
                    nonlocal task_progress
                    task_progress[name] += 1
                    print(
                        f"\rWorking on {name}: {task_progress[name]}/{tasks[name]}",
                        end="",
                    )
                    main_bar()

                return update

            result[task_name] = make_updater(task_name)

        # Add cleanup function
        def cleanup():
            """Clean up the progress bar."""
            print()  # Add a newline after the last task update
            progress_bar.__exit__(None, None, None)

        result["_cleanup"] = cleanup

        return result


# Backward compatibility functions
def create_progress_bar(
    total: Optional[int] = None,
    title: Optional[str] = None,
    spinner: str = None,
    bar: str = None,
    theme: str = None,
    enrich_print: bool = True,
    manual: bool = False,
    stats: bool = True,
    title_color: str = None,
    **kwargs,
) -> Callable:
    """
    Create an enhanced progress bar with customized appearance.

    This function provides backward compatibility with the previous API.
    It is recommended to use the ProgressBar class directly for new code.

    Args:
        total: The total number of items to process.
        title: The title of the progress bar.
        spinner: The spinner style to use.
        bar: The bar style to use.
        theme: The theme to use.
        enrich_print: Whether to enrich print statements.
        manual: Whether to use manual mode.
        stats: Whether to show stats.
        title_color: The color for the title.
        **kwargs: Additional arguments to pass to alive_bar.

    Returns:
        A callable that creates a progress bar context manager.
    """
    progress_bar = ProgressBar(
        total=total,
        title=title,
        spinner=spinner,
        bar=bar,
        theme=theme,
        enrich_print=enrich_print,
        manual=manual,
        stats=stats,
        title_color=title_color,
        **kwargs,
    )

    return progress_bar.__enter__


def process_with_progress(
    items: Iterable[T],
    process_func: Callable[[T], Any],
    title: Optional[str] = None,
    **kwargs,
) -> List[Any]:
    """
    Process a collection of items with a progress bar.

    This function provides backward compatibility with the previous API.
    It is recommended to use the ProgressBar class directly for new code.

    Args:
        items: The items to process.
        process_func: The function to apply to each item.
        title: The title for the progress bar.
        **kwargs: Additional arguments to pass to create_progress_bar.

    Returns:
        A list of processed items.
    """
    # Try to get the length of items for the progress bar
    try:
        total = len(items)
    except (TypeError, AttributeError):
        # If items doesn't support len(), use None for unknown total
        total = None

    progress_bar = ProgressBar(total=total, title=title, **kwargs)
    return progress_bar.process_items(items, process_func)


def timed_progress(seconds: int, title: str = "Processing", step: float = 0.1, **kwargs) -> None:
    """
    Create a progress bar that runs for a specified number of seconds.

    This function provides backward compatibility with the previous API.
    It is recommended to use the ProgressBar class directly for new code.

    Args:
        seconds: The number of seconds to run.
        title: The title for the progress bar.
        step: The time step in seconds.
        **kwargs: Additional arguments to pass to create_progress_bar.
    """
    ProgressBar.timed(seconds, title, step, **kwargs)


def create_multi_bar(tasks: Dict[str, int], **kwargs) -> Dict[str, Callable]:
    """
    Create a simulated multi-progress bar system.

    This function provides backward compatibility with the previous API.
    It is recommended to use the ProgressBar class directly for new code.

    Note: alive_progress doesn't support nested progress bars, so this function
    provides a simpler alternative that uses a single progress bar with a title
    that changes based on the current task.

    Args:
        tasks: A dictionary mapping task names to their total steps.
        **kwargs: Additional arguments to pass to create_progress_bar.

    Returns:
        A dictionary mapping task names to their progress bar update functions.
    """
    return ProgressBar.multi_task(tasks, **kwargs)


def available_spinners() -> List[str]:
    """
    Get a list of all available spinner styles.

    This function provides backward compatibility with the previous API.
    It is recommended to use the ProgressBar class directly for new code.

    Returns:
        A list of spinner style names.
    """
    return ProgressBar.get_available_spinners()


def available_bars() -> List[str]:
    """
    Get a list of all available bar styles.

    This function provides backward compatibility with the previous API.
    It is recommended to use the ProgressBar class directly for new code.

    Returns:
        A list of bar style names.
    """
    return ProgressBar.get_available_bars()
