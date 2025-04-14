"""
Registry mechanism for managing models, losses, metrics, and other components.
"""

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from pydantic import BaseModel

from utils.logger import ColorLogger

logger = ColorLogger(name="Register")


class Register:
    """A registry for functions or classes."""

    def __init__(self, registry_name: str):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key: str, value: Callable) -> None:
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        # Always use the class/function name if key is None or empty
        if not key:
            key = value.__name__
        if key in self._dict:
            logger.warning(f"Key {key} already in registry {self._name}.")
        self._dict[key] = value

    def register(self, target: Union[str, Callable]) -> Callable:
        """
        Register a callable (class or function).

        Args:
            target: Either a callable to register, or a string key to register the callable under.
                   If None or empty string is provided, the callable's name will be used.
        """

        def add_to_registry(key: Optional[str], value: Callable) -> Callable:
            # Always use the class name if key is None or empty
            if not key:
                key = value.__name__
            self[key] = value
            return value

        if callable(target):
            # If target is a callable, use its name as the key
            return add_to_registry(None, target)

        # If target is a string, use it as the key
        return lambda x: add_to_registry(target, x)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def keys(self) -> List[str]:
        return list(self._dict.keys())

    def items(self) -> List[Tuple[str, Any]]:
        return list(self._dict.items())


class Registers:
    """Container for all registries."""

    def __init__(self):
        raise RuntimeError("Registers is not intended to be instantiated")

    model_registry = Register("model")
    loss_registry = Register("loss")
    metric_registry = Register("metric")
    scheduler_registry = Register("scheduler")
    optimizer_registry = Register("optimizer")
    model_config_registry = Register("model_config")
    trainer_config_registry = Register("trainer_config")
    dataset_registry = Register("dataset")
    train_script_registry = Register("train_script")


def build_from_config(config: Union[Dict, Any], registry: Register) -> Any:
    """Build an object from configuration using the specified registry.

    Args:
        config: Either a dictionary or a config object
        registry: The registry to build from

    Returns:
        The constructed object
    """
    # If it is a configuration class instance, convert it to a dictionary.
    if isinstance(config, BaseModel):
        config_dict = config.model_dump()
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise TypeError(
            f"Config must be either a dictionary or a dataclass instance, got {type(config)}"
        )
    # Get type and remove from configuration
    obj_type = config_dict.pop("Name", None)
    if obj_type is None:
        raise ValueError("Config must contain a 'Name' field")

    if obj_type not in registry:
        raise KeyError(f"Component {obj_type} not found in registry {registry._name}")

    # Retrieve class and build an instance
    obj_cls = registry[obj_type]
    try:
        return obj_cls(**config_dict)
    except TypeError as e:
        logger.error(f"Failed to instantiate {obj_type} with config: {config_dict}")
        raise TypeError(f"Configuration error for {obj_type}: {str(e)}")
