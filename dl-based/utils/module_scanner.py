"""
Module scanner and interactive selection functionality.
"""

import importlib
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from colorama import Fore
from colorama import Style
from utils.logger import ColorLogger

logger = ColorLogger(name="ModuleScanner")


class ModuleCategory:
    """Module category configuration"""

    def __init__(self, name: str, directory: str, color: str = Fore.CYAN):
        self.name = name
        self.directory = directory
        self.modules = []
        self.color = color

    def __str__(self):
        return f"{self.color}{self.name}{Style.RESET_ALL} ({len(self.modules)} modules)"


# Define module categories with different colors
MODULE_CATEGORIES = [
    ModuleCategory("model", "models", Fore.CYAN),
    ModuleCategory("loss", "loss", Fore.GREEN),
    ModuleCategory("metric", "metrics", Fore.YELLOW),
    ModuleCategory("optimizer", "optimizer", Fore.MAGENTA),
    ModuleCategory("scheduler", "scheduler", Fore.BLUE),
    ModuleCategory("config", "config", Fore.RED),
    ModuleCategory("dataset", "datasets", Fore.LIGHTCYAN_EX),
]


def scan_modules() -> None:
    """Automatically scan for available modules in each category."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for category in MODULE_CATEGORIES:
        category_path = os.path.join(base_path, category.directory)
        if not os.path.exists(category_path):
            continue

        for file in os.listdir(category_path):
            if file.endswith(".py") and not file.startswith("__"):
                module_name = file[:-3]  # Remove .py extension
                category.modules.append(module_name)


def get_available_components() -> Dict[str, List[str]]:
    """Get all available components by category."""
    components = {}
    for category in MODULE_CATEGORIES:
        if category.modules:
            components[category.name] = category.modules
    return components


def interactive_select(
    category: ModuleCategory, components: List[str]
) -> Optional[str]:
    """Provide interactive selection for components with colored output."""
    if not components:
        logger.warning(f"No {category.name} available.")
        return None

    print(
        f"\n{Fore.WHITE}{Style.BRIGHT}Available {category.color}{category.name}{Style.RESET_ALL}:"
    )

    for i, component in enumerate(components, 1):
        print(
            f"{Fore.LIGHTWHITE_EX}{i}.{Style.RESET_ALL} {Fore.YELLOW}{component}{Style.RESET_ALL}"
        )

    while True:
        try:
            choice = input(
                f"\n{Fore.WHITE}Select {category.name.lower()} ({Fore.GREEN}1-{len(components)}{Fore.WHITE}, or {Fore.RED}0{Fore.WHITE} to skip): {Style.RESET_ALL}"
            )
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(components):
                return components[idx]
            print(f"{Fore.RED}Invalid selection. Please try again.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")


def _handle_errors(errors: List[Tuple[str, Exception]]) -> None:
    """Handle import errors with detailed information."""
    if not errors:
        return

    logger.error("Module import errors occurred:")
    for name, err in errors:
        logger.warning(f"Failed to import module: {name}")
        logger.warning(f"Error type: {type(err).__name__}")
        logger.warning(f"Error message: {str(err)}")
        if hasattr(err, "__traceback__"):
            import traceback

            tb_str = "".join(traceback.format_tb(err.__traceback__))
            logger.warning(f"Traceback:\n{tb_str}")
    logger.error(
        "Please check these modules and ensure they exist and are properly implemented."
    )


def import_modules(interactive: bool = False) -> Dict[str, str]:
    """Import all modules with optional interactive selection."""
    try:
        scan_modules()
    except Exception as e:
        logger.error(f"Error during module scanning: {str(e)}")
        return {}

    selected_components = {}

    if interactive:
        print(
            f"\n{Fore.WHITE}{Style.BRIGHT}{'='*20} Component Selection {'='*20}{Style.RESET_ALL}"
        )

        for category in MODULE_CATEGORIES:
            try:
                if category.modules:
                    selected = interactive_select(category, category.modules)
                    if selected:
                        selected_components[category.name] = selected
            except Exception as e:
                logger.error(
                    f"Error during interactive selection for {category.name}: {str(e)}"
                )

        print(f"\n{Fore.WHITE}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")

    errors = []
    for category in MODULE_CATEGORIES:
        modules_to_import = (
            [selected_components.get(category.name)]
            if interactive and category.name in selected_components
            else category.modules
        )

        for name in modules_to_import:
            if name:
                try:
                    full_name = f"{category.directory}.{name}"
                    logger.debug(f"Attempting to import: {full_name}")
                    importlib.import_module(full_name)
                    logger.debug(
                        f"Module {category.color}{full_name}{Style.RESET_ALL} loaded."
                    )
                except ImportError as error:
                    errors.append((name, error))
                except Exception as e:
                    logger.error(f"Unexpected error while importing {name}: {str(e)}")
                    errors.append((name, e))

    _handle_errors(errors)
    return selected_components
