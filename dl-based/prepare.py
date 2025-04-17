import torch

from typing import Any
from typing import Dict
from utils import ColorLogger
from utils import Registers
from utils.display import pretty_dict
from torchinsight import analyze_model

logger = ColorLogger(name="Prepare-Ctr-Train")


def setup_components(selected: Dict) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Set up all training components based on selected configurations."""
    # Create a dataset
    dataset = Registers.dataset_registry[selected["dataset"]](
        data_path="../data/criteo_data.parquet"
    )

    # Load configurations
    feature_dims = dataset.feature_dims
    dense_feature_dim = dataset.dense_feature_dim
    model_cfg = Registers.model_config_registry[selected["model_config"]](
        feature_dims=feature_dims, dense_feature_dim=dense_feature_dim
    )
    trainer_cfg = Registers.trainer_config_registry[selected["trainer_config"]]().to_dict()
    train_batch_size = trainer_cfg["train_batch_size"]
    test_batch_size = trainer_cfg["test_batch_size"]

    # Dataloader
    train_loader = dataset.get_dataloader(
        dataset.train_dataset, batch_size=train_batch_size, shuffle=True
    )
    val_loader = dataset.get_dataloader(
        dataset.val_dataset, batch_size=test_batch_size, shuffle=False
    )

    # Model
    model = Registers.model_registry[selected["model"]].from_config(model_cfg)

    # Analyze model structure and parameters
    logger.info("Analyzing model structure and parameters...")
    print(
        analyze_model(
            model,
            model_name=model.name if hasattr(model, "name") else selected["model"],
            input_dims=[(dense_feature_dim,), (len(feature_dims),)],
            long_indices=[1],
            batch_size=train_batch_size,
        )
    )

    # Optimizer
    optimizer_cfg = trainer_cfg.pop("optimizer")
    optimizer = Registers.optimizer_registry[selected["optimizer"]].from_config(
        model.parameters(), optimizer_cfg
    )
    logger.info(f"Optimizer:")
    pretty_dict(optimizer_cfg, title=f"{optimizer.name} Optimizer Config")

    # Scheduler
    scheduler_cfg = trainer_cfg.pop("scheduler")
    scheduler = Registers.scheduler_registry[selected["scheduler"]].from_config(
        scheduler_cfg, optimizer
    )
    logger.info(f"Scheduler:")
    pretty_dict(scheduler_cfg, title=f"{scheduler.name} Scheduler Config")

    # Loss
    loss_fn = Registers.loss_registry[selected["loss"]]()
    logger.info(f"Loss Function: {loss_fn.name}")

    # Criterion
    criterion = Registers.metric_registry[selected["metric"]]()
    logger.info(f"Criterion: {criterion.name}")

    # train_script
    if selected["train_script"] == "SingleGPUTrainScript":
        trainer_cfg["Name"] = "SingleGPUTrainScript"
        trainer_cfg["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = Registers.train_script_registry[selected["train_script"]].from_config(trainer_cfg)

    # Create project_info dictionary with non-None values
    project_info = {}

    # Add model name and parameters
    project_info["model_name"] = model.name if hasattr(model, "name") else selected["model"]

    # Add model configuration
    if model_cfg:
        if hasattr(model_cfg, "model_dump"):
            model_params = {k: v for k, v in model_cfg.model_dump().items() if v is not None}
        else:
            model_params = {
                k: v
                for k, v in model_cfg.__dict__.items()
                if not k.startswith("_") and v is not None
            }
        project_info["model_params"] = model_params

    # Add trainer configuration
    if trainer_cfg:
        project_info["trainer_cfg"] = {k: v for k, v in trainer_cfg.items() if v is not None}

    # Add optimizer configuration
    if optimizer_cfg:
        project_info["optimizer_cfg"] = {k: v for k, v in optimizer_cfg.items() if v is not None}

    # Add scheduler configuration
    if scheduler_cfg:
        project_info["scheduler_cfg"] = {k: v for k, v in scheduler_cfg.items() if v is not None}

    # Return both components and project_info
    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "scheduler": scheduler,
        "criterion": criterion,
        "trainer": trainer,
    }, project_info
