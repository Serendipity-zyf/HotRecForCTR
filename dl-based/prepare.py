import torch.nn as nn

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
    if selected["dataset"] == "CriteoDataset":
        dataset = Registers.dataset_registry[selected["dataset"]](
            data_path="../data/criteo/criteo_data_with_interactions.parquet",
            scaled="LogScaler",
        )
    elif selected["dataset"] == "AmazonDataset":
        train_data_path = "../data/amazon/train/train_data.txt"
        test_data_path = "../data/amazon/test/test_data.txt"
        user_vocab_path = "../data/amazon/statistic/user_vocab.json"
        item_vocab_path = "../data/amazon/statistic/item_vocab.json"
        category_vocab_path = "../data/amazon/statistic/category_vocab.json"
        item_info_path = "../data/amazon/item/item_info.pkl"
        train_dataset = Registers.dataset_registry[selected["dataset"]](
            data_path=train_data_path,
            user_vocab_path=user_vocab_path,
            item_vocab_path=item_vocab_path,
            category_vocab_path=category_vocab_path,
            item_info_path=item_info_path,
            scaled="LogScaler",
        )
        test_dataset = Registers.dataset_registry[selected["dataset"]](
            data_path=test_data_path,
            user_vocab_path=user_vocab_path,
            item_vocab_path=item_vocab_path,
            category_vocab_path=category_vocab_path,
            uid_vocab=train_dataset.uid_vocab,
            item_vocab=train_dataset.item_vocab,
            category_vocab=train_dataset.category_vocab,
            item_info_path=item_info_path,
            scaled="LogScaler",
        )

    # Load configurations
    if selected["dataset"] == "CriteoDataset":
        feature_dims = dataset.feature_dims
        dense_feature_dims = dataset.dense_feature_dims
        interact_feature_nums = dataset.interact_feature_nums
        model_cfg = Registers.model_config_registry[selected["model_config"]](
            feature_dims=feature_dims,
            dense_feature_dims=dense_feature_dims,
            interact_feature_nums=interact_feature_nums,
        )
    trainer_cfg = Registers.trainer_config_registry[selected["trainer_config"]]().to_dict()
    train_batch_size = trainer_cfg["train_batch_size"]
    test_batch_size = trainer_cfg["test_batch_size"]

    # Dataloader
    if selected["dataset"] == "CriteoDataset":
        train_loader = dataset.get_dataloader(
            dataset.train_dataset, batch_size=train_batch_size, shuffle=True
        )
        val_loader = dataset.get_dataloader(dataset.val_dataset, batch_size=test_batch_size, shuffle=False)
    elif selected["dataset"] == "AmazonDataset":
        train_loader = train_dataset.get_dataloader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = test_dataset.get_dataloader(test_dataset, batch_size=test_batch_size, shuffle=False)
    # Model
    model = Registers.model_registry[selected["model"]].from_config(model_cfg)

    # Analyze model structure and parameters
    logger.info("Analyzing model structure and parameters...")
    if selected["dataset"] == "CriteoDataset":
        print(
            analyze_model(
                model,
                model_name=model.name if hasattr(model, "name") else selected["model"],
                input_dims=[(dense_feature_dims,), (len(feature_dims),)],
                long_indices=[1],
                device=trainer_cfg["device"],
                batch_size=train_batch_size,
            )
        )
    # Optimizer
    optimizer_cfg = trainer_cfg.pop("optimizer")
    embed_weight_decay = optimizer_cfg.pop("embed_weight_decay", 1e-6)
    dense_weight_decay = optimizer_cfg.pop("dense_weight_decay", 5e-5)
    model_params = get_param_groups_by_type(
        model, embed_weight_decay=embed_weight_decay, dense_weight_decay=dense_weight_decay
    )
    optimizer = Registers.optimizer_registry[selected["optimizer"]].from_config(model_params, optimizer_cfg)
    logger.info(f"Optimizer:")
    pretty_dict(
        {
            **optimizer_cfg,
            "embed_weight_decay": embed_weight_decay,
            "dense_weight_decay": dense_weight_decay,
        },
        title=f"{optimizer.name} Optimizer Config",
    )

    # Scheduler
    scheduler_cfg = trainer_cfg.pop("scheduler")
    scheduler = Registers.scheduler_registry[selected["scheduler"]].from_config(scheduler_cfg, optimizer)
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
        # trainer_cfg["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                k: v for k, v in model_cfg.__dict__.items() if not k.startswith("_") and v is not None
            }
        project_info["model_params"] = model_params

    # Add trainer configuration
    if trainer_cfg:
        project_info["trainer_cfg"] = {k: v for k, v in trainer_cfg.items() if v is not None}

    # Add optimizer configuration
    if optimizer_cfg:
        project_info["optimizer_cfg"] = {k: v for k, v in optimizer_cfg.items() if v is not None}
        project_info["optimizer_cfg"]["embed_weight_decay"] = embed_weight_decay
        project_info["optimizer_cfg"]["dense_weight_decay"] = dense_weight_decay

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


def get_param_groups_by_type(
    model, embed_weight_decay: float, dense_weight_decay: float
) -> list[Dict[str, Any]]:
    embedding_params = set()
    other_params = set()

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for param in module.parameters():
                embedding_params.add(param)

    for param in model.parameters():
        if param not in embedding_params:
            other_params.add(param)

    return [
        {"params": list(embedding_params), "weight_decay": embed_weight_decay},
        {"params": list(other_params), "weight_decay": dense_weight_decay},
    ]
