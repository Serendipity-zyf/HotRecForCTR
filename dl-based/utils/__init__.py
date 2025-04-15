"""
Utility modules for the deep learning-based recommendation system.
"""

from .logger import ColorLogger
from .types import ModuleCategory
from .interactive_selector import interactive_select
from .module_scanner import MODULE_CATEGORIES
from .module_scanner import import_modules
from .progress import ProgressBar
from .progress import create_progress_bar
from .register import Register
from .register import Registers
from .register import build_from_config
from .display import display_dict
from .display import display_nested_dict
from .display import pretty_dict


__all__ = [
    "ColorLogger",
    "create_progress_bar",
    "ProgressBar",
    "Register",
    "Registers",
    "build_from_config",
    "interactive_select",
    "import_modules",
    "MODULE_CATEGORIES",
    "ModuleCategory",
    "display_dict",
    "display_nested_dict",
    "pretty_dict",
]
