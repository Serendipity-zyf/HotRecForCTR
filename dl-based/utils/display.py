"""
Utility functions for displaying data in a formatted way.
"""

import prettytable

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from colorama import Fore
from colorama import Style

# Type alias for dictionary values
DictValue = Union[str, int, float, bool, List, Tuple, Dict, None]


def display_dict(
    data: Dict[str, DictValue],
    title: str = "Dictionary Contents",
    key_color: str = Fore.CYAN,
    value_color: str = Fore.YELLOW,
    title_color: str = Fore.GREEN + Style.BRIGHT,
    header_color: str = Fore.WHITE + Style.BRIGHT,
    border_color: str = Fore.BLUE,
    max_width: int = 80,
    sort_keys: bool = True,
) -> None:
    """
    Display a dictionary in a formatted table with color support.

    This function handles various value types including tuples, lists, and nested dictionaries.

    Args:
        data: The dictionary to display
        title: The title of the table
        key_color: Color for the keys
        value_color: Color for the values
        title_color: Color for the title
        header_color: Color for the table headers
        border_color: Color for the table borders
        max_width: Maximum width for value display
        sort_keys: Whether to sort the keys alphabetically
    """
    if not data:
        print(f"{title_color}{title}{Style.RESET_ALL}")
        print(f"{Fore.RED}Empty dictionary{Style.RESET_ALL}")
        return

    # Create a PrettyTable instance
    table = prettytable.PrettyTable()

    # Set field names with colors
    table.field_names = [
        f"{header_color}Key{Style.RESET_ALL}",
        f"{header_color}Value{Style.RESET_ALL}",
        f"{header_color}Type{Style.RESET_ALL}",
    ]

    # Set alignment
    table.align["Key"] = "l"
    table.align["Value"] = "l"
    table.align["Type"] = "l"

    # Set max width for the value column
    table.max_width["Value"] = max_width

    # Get the keys and sort them if requested
    keys = list(data.keys())
    if sort_keys:
        keys.sort()

    # Add rows to the table
    for key in keys:
        value = data[key]
        value_type = type(value).__name__

        # Format the value based on its type
        if isinstance(value, (list, tuple)):
            if len(value) > 5:
                formatted_value = f"{value[:5]} ... (total: {len(value)} items)"
            else:
                formatted_value = str(value)
        elif isinstance(value, dict):
            if len(value) > 3:
                keys_preview = list(value.keys())[:3]
                formatted_value = f"{{{', '.join(str(k) for k in keys_preview)}, ... }} (total: {len(value)} items)"
            else:
                formatted_value = str(value)
        else:
            formatted_value = str(value)

        # Truncate long values
        if len(formatted_value) > max_width:
            formatted_value = formatted_value[: max_width - 3] + "..."

        # Add the row with colors
        table.add_row(
            [
                f"{key_color}{key}{Style.RESET_ALL}",
                f"{value_color}{formatted_value}{Style.RESET_ALL}",
                f"{Fore.MAGENTA}{value_type}{Style.RESET_ALL}",
            ]
        )

    # Set border color
    table_str = table.get_string()
    colored_table = ""
    for line in table_str.split("\n"):
        if any(c in line for c in "+-|"):
            # Apply border color to lines with borders
            colored_line = border_color + line + Style.RESET_ALL
            colored_table += colored_line + "\n"
        else:
            colored_table += line + "\n"

    # Print the title and table
    print(f"\n{title_color}{title}{Style.RESET_ALL}")
    print(colored_table)


def display_nested_dict(
    data: Dict[str, Any],
    title: str = "Nested Dictionary Contents",
    indent: int = 0,
    key_colors: List[str] = [Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.MAGENTA],
    value_color: str = Fore.WHITE,
    title_color: str = Fore.GREEN + Style.BRIGHT,
    max_items: int = 10,
    max_depth: int = 3,
) -> None:
    """
    Display a nested dictionary with indentation and color coding by level.

    Args:
        data: The nested dictionary to display
        title: The title to display
        indent: Current indentation level (used in recursion)
        key_colors: List of colors to use for keys at different levels
        value_color: Color for values
        title_color: Color for the title
        max_items: Maximum number of items to display at each level
        max_depth: Maximum depth to display
    """
    if indent == 0:
        print(f"\n{title_color}{title}{Style.RESET_ALL}")

    if not data:
        print(f"{' ' * indent * 2}{Fore.RED}Empty dictionary{Style.RESET_ALL}")
        return

    if indent >= max_depth:
        print(f"{' ' * indent * 2}{Fore.RED}... (max depth reached){Style.RESET_ALL}")
        return

    # Get the current level's key color
    key_color = key_colors[min(indent, len(key_colors) - 1)]

    # Get the keys and sort them
    keys = sorted(data.keys())

    # Check if we need to limit the number of items
    if len(keys) > max_items:
        display_keys = keys[:max_items]
        has_more = True
    else:
        display_keys = keys
        has_more = False

    # Display each key-value pair
    for key in display_keys:
        value = data[key]
        indent_str = " " * indent * 2

        if isinstance(value, dict):
            print(f"{indent_str}{key_color}{key}{Style.RESET_ALL}:")
            display_nested_dict(
                value,
                indent=indent + 1,
                key_colors=key_colors,
                value_color=value_color,
                max_items=max_items,
                max_depth=max_depth,
            )
        elif isinstance(value, (list, tuple)):
            if len(value) > 5:
                print(
                    f"{indent_str}{key_color}{key}{Style.RESET_ALL}: {value_color}{value[:5]} ... (total: {len(value)} items){Style.RESET_ALL}"
                )
            else:
                print(
                    f"{indent_str}{key_color}{key}{Style.RESET_ALL}: {value_color}{value}{Style.RESET_ALL}"
                )
        else:
            print(
                f"{indent_str}{key_color}{key}{Style.RESET_ALL}: {value_color}{value}{Style.RESET_ALL}"
            )

    # Indicate if there are more items
    if has_more:
        print(
            f"{' ' * indent * 2}{Fore.RED}... ({len(keys) - max_items} more items){Style.RESET_ALL}"
        )


def pretty_dict(
    data: Dict[str, Any], title: str = None, table_format: bool = True, **kwargs
) -> None:
    """
    Display a dictionary in a pretty format.

    This is a convenience function that chooses between display_dict and
    display_nested_dict based on the table_format parameter.

    Args:
        data: The dictionary to display
        title: The title to display (defaults to "Dictionary Contents" or "Nested Dictionary Contents")
        table_format: Whether to use table format (True) or nested format (False)
        **kwargs: Additional arguments to pass to the display function
    """
    if table_format:
        if title is None:
            title = "Dictionary Contents"
        display_dict(data, title=title, **kwargs)
    else:
        if title is None:
            title = "Nested Dictionary Contents"
        display_nested_dict(data, title=title, **kwargs)
