"""
Main script for deep learning-based CTR prediction.
"""

from utils import ColorLogger
from utils import Registers
from utils import import_modules
from alive_progress import config_handler

logger = ColorLogger(name="CTRTrainScript")


def setup_components(selected):
    """Set up all training components based on selected configurations."""
    # Create a dataset
    dataset = Registers.dataset_registry[selected["dataset"]](
        data_path="../data/criteo_data.parquet"
    )

    # Load configurations
    model_cfg = Registers.model_config_registry[selected["model_config"]](
        feature_dims=dataset.feature_dims, dense_feature_dim=dataset.dense_feature_dim
    )
    trainer_cfg = Registers.trainer_config_registry[
        selected["trainer_config"]
    ]().model_dump()
    train_batch_size = trainer_cfg["train_batch_size"]
    test_batch_size = trainer_cfg["test_batch_size"]

    # Obtain training and validation dataloaders
    train_loader = dataset.get_dataloader(
        dataset.train_dataset, batch_size=train_batch_size, shuffle=True
    )
    val_loader = dataset.get_dataloader(
        dataset.val_dataset, batch_size=test_batch_size, shuffle=False
    )

    # Model load
    model = Registers.model_registry[selected["model"]].from_config(model_cfg)
    logger.info(f"Model Structure:\n {model}")

    # Optimizer
    optimizer_cfg = trainer_cfg.pop("optimizer")
    optimizer = Registers.optimizer_registry[selected["optimizer"]].from_config(
        model.parameters(), optimizer_cfg
    )
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Optimizer Config: {optimizer_cfg}")

    # Scheduler
    scheduler_cfg = trainer_cfg.pop("scheduler")
    scheduler = Registers.scheduler_registry[selected["scheduler"]].from_config(
        scheduler_cfg, optimizer
    )
    logger.info(f"Scheduler: {scheduler}")
    logger.info(f"Scheduler Config: {scheduler_cfg}")

    # Loss function
    loss_fn = Registers.loss_registry[selected["loss"]]()
    logger.info(f"Loss Function: {loss_fn}")

    # Criterion
    criterion = Registers.metric_registry[selected["metric"]]()
    logger.info(f"Criterion: {criterion}")

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": loss_fn,
        "criterion": criterion,
        "trainer_cfg": trainer_cfg,
    }


def main():
    # Import modules with interactive selection
    selected = import_modules(interactive=True)

    # Set up all components
    components = setup_components(selected)

    # Show selected components
    logger.info("Selected components:")
    for category, component in selected.items():
        print(f"{category}: {component}")

    # TODO: Initialize trainer and start training


if __name__ == "__main__":
    main()
