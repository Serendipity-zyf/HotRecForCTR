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
    dataset = Registers.dataset_registry[selected["dataset"]](
        data_path="../data/criteo_data.parquet"
    )

    # Load configurations
    model_cfg = Registers.model_config_registry[selected["model_config"]](
        feature_dims=dataset.feature_dims, dense_feature_dim=dataset.dense_feature_dim
    )
    trainer_cfg = Registers.trainer_config_registry[selected["trainer_config"]]()
    train_batch_size = trainer_cfg.train_batch_size
    test_batch_size = trainer_cfg.test_batch_size

    # Obtain training and validation dataloaders
    train_loader = dataset.get_dataloader(
        dataset.train_dataset, batch_size=train_batch_size, shuffle=True
    )

    val_loader = dataset.get_dataloader(
        dataset.val_dataset, batch_size=test_batch_size, shuffle=False
    )

    # model load
    model = Registers.model_registry[selected["model"]].from_config(model_cfg)
    logger.info(f"Model Structure:\n {model}")


if __name__ == "__main__":
    main()
