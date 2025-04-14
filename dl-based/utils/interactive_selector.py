"""
Interactive selection functionality for components.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

from colorama import Fore, Style
from .logger import ColorLogger
from .types import ModuleCategory

logger = ColorLogger(name="InteractiveSelector")


class SelectionResult(Enum):
    """Enumeration for selection results"""

    VALID = "valid"
    INVALID = "invalid"
    SKIP = "skip"


@dataclass
class Selection:
    """Data class for selection result"""

    result: SelectionResult
    value: Optional[int] = None
    message: Optional[str] = None


def validate_selection(
    choice: str, components_len: int, allow_skip: bool = True
) -> Selection:
    """Validate user selection input."""
    try:
        value = int(choice)
        if value == 0:
            if not allow_skip:
                return Selection(
                    SelectionResult.INVALID,
                    message="This component cannot be skipped. Please make a selection.",
                )
            return Selection(SelectionResult.SKIP)
        if 1 <= value <= components_len:
            return Selection(SelectionResult.VALID, value=value - 1)
        return Selection(
            SelectionResult.INVALID,
            message=f"Please enter a number between 1 and {components_len}",
        )
    except ValueError:
        return Selection(SelectionResult.INVALID, message="Please enter a valid number")


def get_selection_prompt(
    category: ModuleCategory, components_len: int, force_select: bool
) -> str:
    """Generate the appropriate selection prompt."""
    if force_select:
        return (
            f"\n{Fore.WHITE}Select {category.name.lower()} "
            f"({Fore.GREEN}1{Fore.WHITE}): {Style.RESET_ALL}"
        )
    return (
        f"\n{Fore.WHITE}Select {category.name.lower()} "
        f"({Fore.GREEN}1-{components_len}{Fore.WHITE}, "
        f"or {Fore.RED}0{Fore.WHITE} to skip): {Style.RESET_ALL}"
    )


def interactive_select(
    category: ModuleCategory, components: List[str]
) -> Optional[str]:
    """Provide interactive selection for components with colored output."""
    if not components:
        logger.warning(f"No {category.name} available.")
        return None

    # Display available components
    print(
        f"\n{Fore.WHITE}{Style.BRIGHT}Available {category.color}{category.name}{Style.RESET_ALL}:"
    )
    for i, component in enumerate(components, 1):
        print(
            f"{Fore.LIGHTWHITE_EX}{i}.{Style.RESET_ALL} {Fore.YELLOW}{component}{Style.RESET_ALL}"
        )

    # 如果只有一个选项，仍然允许跳过
    force_select = False

    while True:
        prompt = get_selection_prompt(category, len(components), force_select)
        choice = input(prompt)

        selection = validate_selection(choice, len(components), not force_select)

        match selection.result:
            case SelectionResult.VALID:
                return components[selection.value]
            case SelectionResult.SKIP:
                return None
            case SelectionResult.INVALID:
                print(f"{Fore.RED}{selection.message}{Style.RESET_ALL}")
