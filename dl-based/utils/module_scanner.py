"""
Module scanner functionality.
"""

import os
import importlib

from typing import Dict
from typing import List
from typing import Tuple
from colorama import Fore
from colorama import Style
from .logger import ColorLogger
from .types import ModuleCategory
from .interactive_selector import interactive_select

logger = ColorLogger(name="ModuleScanner")


# Define module categories with different colors
MODULE_CATEGORIES = [
    ModuleCategory("model", "models", Fore.CYAN),
    ModuleCategory("loss", "loss", Fore.GREEN),
    ModuleCategory("metric", "metrics", Fore.YELLOW),
    ModuleCategory("optimizer", "optimizer", Fore.MAGENTA),
    ModuleCategory("scheduler", "scheduler", Fore.BLUE),
    ModuleCategory("model_config", "config", Fore.RED),
    ModuleCategory("trainer_config", "config", Fore.LIGHTBLUE_EX),
    ModuleCategory("dataset", "datasets", Fore.LIGHTCYAN_EX),
    ModuleCategory("train_script", "train_scripts", Fore.LIGHTGREEN_EX),
]


def is_valid_module(filename: str) -> bool:
    """Check if the file is a valid module to import."""
    return (
        filename.endswith(".py")
        and not filename.startswith("__")
        and "base" not in filename.lower()
    )


def scan_modules() -> None:
    """Automatically scan for available modules in each category."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for category in MODULE_CATEGORIES:
        category_path = os.path.join(base_path, category.directory)
        if not os.path.exists(category_path):
            continue

        category.modules.extend(
            [file[:-3] for file in os.listdir(category_path) if is_valid_module(file)]
        )


def get_available_components() -> Dict[str, List[str]]:
    """Get all available components by category."""
    return {
        category.name: category.modules
        for category in MODULE_CATEGORIES
        if category.modules
    }


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


def _get_registered_components(category_name: str) -> List[str]:
    """Get the registered components for a specific category."""
    from .register import Registers

    registry_map = {
        "model_config": Registers.model_config_registry,
        "trainer_config": Registers.trainer_config_registry,
        "model": Registers.model_registry,
        "dataset": Registers.dataset_registry,
        "loss": Registers.loss_registry,
        "metric": Registers.metric_registry,
        "optimizer": Registers.optimizer_registry,
        "scheduler": Registers.scheduler_registry,
        "train_script": Registers.train_script_registry,
    }

    return list(registry_map.get(category_name, {}).keys())


def import_modules(interactive: bool = False) -> Dict[str, str]:
    """Import all modules with optional interactive selection."""
    try:
        scan_modules()
    except Exception as e:
        logger.error(f"Error during module scanning: {str(e)}")
        return {}

    selected_components = {}
    errors = []

    # First, import all modules to register components
    for category in MODULE_CATEGORIES:
        for name in category.modules:
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

    if interactive:
        print(
            f"\n{Fore.WHITE}{Style.BRIGHT}{'='*20} Component Selection {'='*20}{Style.RESET_ALL}"
        )

        for category in MODULE_CATEGORIES:
            try:
                # Always use registered components for all categories
                registered_components = _get_registered_components(category.name)
                if registered_components:
                    selected = interactive_select(category, registered_components)
                    if selected:
                        selected_components[category.name] = selected
                elif category.modules and not registered_components:
                    # Fallback to module names if no registered components found
                    selected = interactive_select(category, category.modules)
                    if selected:
                        selected_components[category.name] = selected
            except Exception as e:
                logger.error(
                    f"Error during interactive selection for {category.name}: {str(e)}"
                )

        print(f"\n{Fore.WHITE}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")

    _handle_errors(errors)
    return selected_components
