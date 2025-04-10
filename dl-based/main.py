"""
Main script for deep learning-based CTR prediction.
"""

from utils import ColorLogger
from utils import Registers
from utils import import_modules

logger = ColorLogger(name="CTRTrainScript")


def main():
    # Import modules with interactive selection
    selected = import_modules(interactive=True)

    # Show selected components
    logger.info("Selected components:")
    for category, component in selected.items():
        print(f"{category}: {component}")

    # Create a dataset
    dataset = Registers.dataset_registry[selected["dataset"]]()

    # Obtain training and validation dataloaders
    train_loader = dataset.get_dataloader(
        dataset.train_dataset, batch_size=128, shuffle=True
    )

    val_loader = dataset.get_dataloader(
        dataset.val_dataset,
        batch_size=256,
        shuffle=False,
    )

    # Log dataset information
    logger.info(f"Training samples: {len(dataset.train_dataset)}")
    logger.info(f"Validation samples: {len(dataset.val_dataset)}")


if __name__ == "__main__":
    main()
