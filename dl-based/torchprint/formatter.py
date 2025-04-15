"""
Formatting utilities for model analysis display.
"""

import math
import re
from typing import Dict
from typing import Tuple
from typing import List
from colorama import Fore, Style

# Define color constants for different parts of the output
TITLE_COLOR = Fore.CYAN + Style.BRIGHT
HEADER_COLOR = Fore.GREEN + Style.BRIGHT
SUMMARY_COLOR = Fore.YELLOW
MEMORY_COLOR = Fore.MAGENTA
SEPARATOR_COLOR = Fore.BLUE
LAYER_NAME_COLOR = Fore.WHITE
MODULE_TYPE_COLOR = Fore.CYAN
PARAM_COLOR = Fore.YELLOW
SHAPE_COLOR = Fore.WHITE
MACS_COLOR = Fore.GREEN
FLOPS_COLOR = Fore.CYAN
TRAINABLE_COLOR = Fore.GREEN
NOT_TRAINABLE_COLOR = Fore.RED


def format_size(size: Tuple[int, ...]) -> str:
    """Format tensor size as a string."""
    if not size:
        return "--"
    return str(list(size))


def format_bytes(num_bytes: int) -> str:
    """Format bytes as a human-readable string with appropriate units."""
    if num_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(num_bytes, 1024)))
    i = min(i, len(units) - 1)  # Cap at the largest unit

    p = math.pow(1024, i)
    s = round(num_bytes / p, 2)

    return f"{s:.2f} {units[i]}"


def format_flops(flops: int) -> str:
    """Format FLOPs as a human-readable string with appropriate units."""
    if flops == 0:
        return "--"

    if flops < 1000:
        return str(flops)

    units = ["", "K", "M", "G", "T"]
    i = 0

    while flops >= 1000 and i < len(units) - 1:
        flops /= 1000
        i += 1

    return f"{flops:.2f} {units[i]}"


def format_param_count(count: int) -> str:
    """Format parameter count as a string."""
    if count == 0:
        return "--"
    return f"{count:,}"


def get_tree_prefix(depth: int, is_last: bool, prefix_dict: Dict[int, bool]) -> str:
    """
    Get the tree prefix for a module at a specific depth.

    Args:
        depth: The depth of the module
        is_last: Whether this module is the last child of its parent
        prefix_dict: Dictionary tracking whether a level has more siblings coming

    Returns:
        Formatted tree prefix string
    """
    if depth == 0:
        return ""

    # Update prefix dictionary
    prefix_dict[depth] = not is_last

    # Build the prefix
    result = ""
    for i in range(1, depth):
        if prefix_dict.get(i, False):
            result += f"{SEPARATOR_COLOR}│{Style.RESET_ALL}   "
        else:
            result += "    "

    if is_last:
        result += f"{SEPARATOR_COLOR}└─{Style.RESET_ALL}"
    else:
        result += f"{SEPARATOR_COLOR}├─{Style.RESET_ALL}"

    return result


def format_layer_name(
    name: str,
    module_type: str,
    depth: int,
    is_last: bool,
    prefix_dict: Dict[int, bool],
    depth_idx: str = "",
) -> str:
    """
    Format layer name with tree structure.

    Args:
        name: Module name
        module_type: Module type
        depth: Module depth
        is_last: Whether this module is the last child of its parent
        prefix_dict: Dictionary tracking whether a level has more siblings coming
        depth_idx: Depth and index identifier (e.g., "1-1")

    Returns:
        Formatted layer name with tree structure
    """
    prefix = get_tree_prefix(depth, is_last, prefix_dict)

    if depth == 0:
        return f"{LAYER_NAME_COLOR}{name}{Style.RESET_ALL}"
    else:
        if depth_idx:
            return f"{prefix}{LAYER_NAME_COLOR}{name}{Style.RESET_ALL}: {MODULE_TYPE_COLOR}{module_type}{Style.RESET_ALL}:{depth_idx}"
        else:
            return f"{prefix}{LAYER_NAME_COLOR}{name}{Style.RESET_ALL}: {MODULE_TYPE_COLOR}{module_type}{Style.RESET_ALL}"


def create_header(col_widths: List[int] = None) -> str:
    """Create the header for the model analysis table."""
    if col_widths is None:
        col_widths = [40, 20, 20, 12, 12, 10]

    layer_col = format_with_color("Layer (type:depth-idx)", col_widths[0], HEADER_COLOR)
    input_col = format_with_color("Input Shape", col_widths[1], HEADER_COLOR)
    output_col = format_with_color("Output Shape", col_widths[2], HEADER_COLOR)
    param_col = format_with_color("Param #", col_widths[3], HEADER_COLOR)
    macs_col = format_with_color("Mult-Adds", col_widths[4], HEADER_COLOR)
    train_col = format_with_color("Trainable", col_widths[5], HEADER_COLOR)

    return layer_col + input_col + output_col + param_col + macs_col + train_col


def create_separator(col_widths: List[int] = None) -> str:
    """Create a separator line for the model analysis table."""
    if col_widths is None:
        col_widths = [40, 20, 20, 12, 12, 10]  # Default widths
    total_width = sum(col_widths)
    return f"{SEPARATOR_COLOR}{'=' * total_width}{Style.RESET_ALL}"


def colorize_trainable(trainable: bool) -> str:
    """Colorize the trainable status."""
    if trainable:
        return f"{TRAINABLE_COLOR}True{Style.RESET_ALL}"
    else:
        return f"{NOT_TRAINABLE_COLOR}False{Style.RESET_ALL}"


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes from a string."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def format_with_color(text: str, width: int, color: str = "") -> str:
    """
    Format text with color and padding to a specific width.

    Args:
        text: The text to format
        width: The desired width
        color: Optional color code to apply to the text

    Returns:
        Formatted text with color
    """
    # Apply color if provided
    if color:
        colored_text = f"{color}{text}{Style.RESET_ALL}"
    else:
        colored_text = text

    # Calculate visible length (without color codes)
    visible_text = strip_ansi(colored_text)
    visible_length = len(visible_text)

    # Add padding to reach the desired width
    padding = max(0, width - visible_length)
    padding_text = " " * padding

    return colored_text + padding_text
