"""Single GPU / CPU training script."""

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from torch.utils.data import DataLoader

from utils.logger import ColorLogger
from utils.register import Registers
from utils.progress import ProgressBar
from utils.register import build_from_config

logger = ColorLogger(name="TrainScriptCriteo")


@Registers.train_script_registry.register
class SingleGPUTrainScriptCriteo(object):
    """Single GPU / CPU training script for Criteo Dataset."""

    def __init__(
        self,
        epochs: int,
        train_batch_size: int,
        test_batch_size: int,
        grad_clip: float,
        is_scheduler: bool,
        patience: int,
        device: str,
    ):
        self.name = "SingleGPUTrainScriptCriteo"
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.grad_clip = grad_clip
        self.is_scheduler = is_scheduler
        self.patience = patience
        self.device = device
        self.train_loss = []
        self.val_loss = []
        self.best_metric = 0
        self.last_metric = 0
        self.early_stop_times = 0
        self.wdb = None

    def setup_wandb(self, wdb: Optional[Any]) -> None:
        """Set up wandb logging."""
        self.wdb = wdb

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
    ) -> None:
        """Train the model.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            loss_fn: Loss function
            scheduler: Learning rate scheduler
            criterion: Metric for model evaluation
        """
        best_model_state = None
        patience = self.patience  # Early stopping patience

        # Use simple log output instead of outer ProgressBar
        logger.info(f"\n{'='*50}\nStarting training, total epochs: {self.epochs}\n{'='*50}")
        for epoch in range(1, self.epochs + 1):
            # Training phase
            model.train()
            train_loss = self._train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epoch=epoch,
            )

            # Validation phase
            model.eval()
            val_loss, metric = self._validate_epoch(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                criterion=criterion,
                epoch=epoch,
            )

            # Early stopping check
            if epoch == 1:
                self.best_metric = metric
                best_model_state = model.state_dict()
            else:
                if metric > self.best_metric:
                    self.best_metric = metric
                    best_model_state = model.state_dict()
                    self.early_stop_times = 0
                else:
                    self.early_stop_times += 1
                    logger.warning(f"Validation metric did not improve for {self.early_stop_times} epochs")

            # Update learning rate if scheduler is enabled
            if self.is_scheduler and scheduler.name == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            elif self.is_scheduler:
                scheduler.step()

            # Use log output instead of progress bar update
            logger.info(
                f"\n{'*'*75}\n"
                f"Epoch [{epoch}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"{criterion.name} Metric: {metric:.4f}\n"
                f"{'*'*75}\n"
            )

            # Early stopping
            if self.early_stop_times >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with metric: {self.best_metric:.4f}")

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        epoch: int,
    ) -> float:
        """Train for one epoch."""
        epoch_losses = []

        # Use ProgressBar for inner loop
        logger.info(f"\n{'-'*35} Training Epoch {epoch} {'-'*35}")
        with ProgressBar(total=len(train_loader), title=f"Training") as bar:
            for batch in train_loader:
                # Move data to device
                dense_x, discrete_x, label = [x.to(self.device) for x in batch]

                # Forward pass
                optimizer.zero_grad()
                output = model(dense_x, discrete_x)
                loss = loss_fn(output, label)

                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

                # Record loss
                current_loss = loss.item()
                epoch_losses.append(current_loss)

                # Update progress bar with current loss
                bar.text = f"[Batch Loss: {current_loss:.4f}]"
                bar()

        # Calculate and record average loss
        avg_loss = sum(epoch_losses) / len(train_loader)
        if self.wdb:
            self.wdb.log({"train_loss": avg_loss}, step=epoch)
        self.train_loss.append(avg_loss)
        logger.info(f"Epoch {epoch} average train loss: {avg_loss:.4f}")
        return avg_loss

    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        criterion: nn.Module,
        epoch: int,
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        val_losses = []
        predictions = []
        targets = []

        # Use ProgressBar for validation loop
        logger.info(f"\n{'-'*35} Validating Epoch {epoch} {'-'*35}")
        with torch.inference_mode(), ProgressBar(total=len(val_loader), title=f"Validation") as bar:
            for batch in val_loader:
                # Move data to device
                dense_x, discrete_x, label = [x.to(self.device) for x in batch]

                # Forward pass
                output = model(dense_x, discrete_x)
                loss = loss_fn(output, label)

                # Record predictions and losses
                predictions.append(output.cpu())
                targets.append(label.cpu())
                val_losses.append(loss.item())

                # Update progress bar with current loss
                bar.text = f"[Batch Loss: {loss.item():.4f}]"
                bar()

        # Calculate metrics
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        avg_loss = sum(val_losses) / len(val_loader)
        metric = criterion(predictions, targets)

        # Record validation loss
        if self.wdb:
            self.wdb.log({"val_loss": avg_loss, f"{criterion.name}": metric}, step=epoch)
        self.val_loss.append(avg_loss)
        logger.info(f"Epoch {epoch} validation - Loss: {avg_loss:.4f}, Metric: {metric:.4f}")

        return avg_loss, metric

    def save(self, model: nn.Module, path: str) -> None:
        """Save the model."""
        torch.save(model.state_dict(), path)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SingleGPUTrainScriptCriteo":
        """Create model from config."""
        return build_from_config(config, Registers.train_script_registry)
