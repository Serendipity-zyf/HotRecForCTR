"""
Main script for deep learning-based CTR prediction.
"""

from utils.logger import ColorLogger
from utils.module_scanner import import_modules
from utils.register import Registers

logger = ColorLogger(name="CTRTrainScript")


def main():
    # Import modules with interactive selection
    selected_components = import_modules(interactive=True)

    # Show selected components
    logger.info("Selected components:")
    for category, component in selected_components.items():
        print(f"{category}: {component}")

    # Show registered models
    logger.info("Registered models:")
    for model_name in Registers.model_registry.keys():
        print(f"- {model_name}")


if __name__ == "__main__":
    main()
