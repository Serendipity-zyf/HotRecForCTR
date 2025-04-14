"""
Model utility functions for analyzing model structure, parameters, and memory usage.
"""

import inspect
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from colorama import Fore, Style
from torchinfo import summary

from .logger import ColorLogger

logger = ColorLogger(name="ModelUtils")


def _create_single_input(
    batch_size: int, input_size: Tuple, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create a single random input tensor for model analysis."""
    return torch.rand(batch_size, *input_size, device=device, dtype=dtype)


def _create_multi_input_tensors(
    model: nn.Module,
    batch_size: int,
    input_sizes: List[Tuple],
    param_names: List[str],
    device: torch.device,
    dtypes: List[torch.dtype],
) -> List[torch.Tensor]:
    """Create input tensors for models with multiple inputs.

    This function creates appropriate tensors for each input parameter.
    By default, all inputs are created as float tensors, which works for most models.
    Only use integer tensors if we explicitly detect they're needed for embeddings.
    """
    # Initialize inputs list
    inputs = [None] * len(param_names)

    # First, check if we need to use integer tensors for any inputs by looking for embedding layers
    has_embeddings = False
    for _, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            has_embeddings = True
            break

    # If we have embeddings, we'll try to intelligently assign dtypes, otherwise use float for all
    if has_embeddings and len(dtypes) < len(param_names):
        # Extend dtypes list if needed
        while len(dtypes) < len(param_names):
            dtypes.append(torch.float32)

        # If we have exactly 2 inputs and one dtype is float and we have embeddings,
        # assume the second input might be categorical
        if len(param_names) == 2 and dtypes[0] == torch.float32:
            dtypes[1] = torch.int64

    # Create appropriate tensor for each input parameter
    for i, input_size in enumerate(input_sizes):
        # Get dtype for this parameter (default to float32)
        param_dtype = dtypes[min(i, len(dtypes) - 1)]

        if param_dtype == torch.int64 or param_dtype == torch.long:
            # For categorical features, create tensor with small random indices
            max_index = 5  # Default small value

            # Look for embedding layers to determine valid index ranges
            embeddings = []
            for _, module in model.named_modules():
                if isinstance(module, nn.Embedding) and hasattr(
                    module, "num_embeddings"
                ):
                    embeddings.append(module)

            # If we have embeddings and this is a categorical input, use appropriate indices
            if embeddings and i < len(embeddings):
                # Use the corresponding embedding's num_embeddings as the max index
                max_index = max(1, embeddings[i].num_embeddings - 1)
            else:
                # Otherwise use a safe default
                max_index = 5

            inputs[i] = torch.randint(
                0,
                max(1, max_index),
                (batch_size, *input_size),
                device=device,
                dtype=param_dtype,
            )
        else:
            # For continuous features, create random float tensor
            inputs[i] = torch.rand(
                batch_size, *input_size, device=device, dtype=param_dtype
            )

    return inputs


def _create_generic_inputs(
    batch_size: int,
    input_sizes: List[Tuple],
    device: torch.device,
    dtypes: List[torch.dtype] = None,
) -> List[torch.Tensor]:
    """Create generic input tensors for models with multiple inputs.

    This is a simpler version that creates all tensors with the same dtype (usually float32).
    For more specialized input creation, use _create_multi_input_tensors.
    """
    if dtypes is None or len(dtypes) == 0:
        dtypes = [torch.float32]

    # For safety, always use float32 for generic inputs unless explicitly specified otherwise
    return [
        torch.rand(batch_size, *size, device=device, dtype=torch.float32)
        for size in input_sizes
    ]


def _get_model_basic_info(model: nn.Module) -> Dict[str, Any]:
    """Calculate basic model information like parameters and size."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "model_size_mb": model_size_mb,
    }


def _print_model_info(
    model: nn.Module,
    device: torch.device,
    model_info: Dict[str, Any],
    title: str,
    title_color: str,
):
    """Print formatted model information."""
    print(f"\n{title_color}{title}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

    # Basic model info
    print(f"{Fore.GREEN}Model Type:{Style.RESET_ALL} {model.__class__.__name__}")
    if hasattr(model, "name"):
        print(f"{Fore.GREEN}Model Name:{Style.RESET_ALL} {model.name}")
    print(f"{Fore.GREEN}Device:{Style.RESET_ALL} {device}")

    # Parameters info
    print(
        f"{Fore.GREEN}Total Parameters:{Style.RESET_ALL} {model_info['total_params']:,}"
    )
    print(
        f"{Fore.GREEN}Trainable Parameters:{Style.RESET_ALL} {model_info['trainable_params']:,}"
    )
    print(
        f"{Fore.GREEN}Non-trainable Parameters:{Style.RESET_ALL} {model_info['non_trainable_params']:,}"
    )

    # Model size
    print(
        f"{Fore.GREEN}Model Size (MB):{Style.RESET_ALL} {model_info['model_size_mb']:.2f}"
    )

    # Torchinfo header
    print(f"\n{Fore.YELLOW}Detailed Model Analysis (torchinfo):{Style.RESET_ALL}")


def analyze_model(
    model: nn.Module,
    input_size: Optional[Union[Tuple, List[Tuple]]] = None,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    col_names: Optional[List[str]] = None,
    verbose: int = 1,
    title: str = "Model Analysis",
    title_color: str = Fore.CYAN + Style.BRIGHT,
) -> Dict[str, Any]:
    """
    Analyze a PyTorch model and display its structure, parameters, and memory usage using torchinfo.

    This function works with any standard PyTorch model structure and automatically adapts to
    different input patterns:
    - Single input models: model(x)
    - Dual input models: model(x1, x2)
    - Multi-input models: model(x1, x2, x3, ...)

    The function does not rely on specific module names or structures and will work with any
    model architecture including attention-based models, convolutional models, or custom designs.

    Args:
        model: The PyTorch model to analyze
        input_size: Size of input tensor(s) to the model. Can be:
            - A tuple for a single input: (features,)
            - A list of tuples for multiple inputs: [(features1,), (features2,)]
            - None to auto-detect input shapes
        batch_size: Batch size for the input tensor(s)
        device: Device to use for analysis (defaults to model's device)
        dtypes: List of dtypes for each input tensor (defaults to [torch.float32, torch.int64])
            For multi-input models, the dtype at index i is used for the i-th input
        col_names: Columns to show in the output table
        verbose: Verbosity level (0-2)
        title: Title for the analysis output
        title_color: Color for the title

    Returns:
        A dictionary containing analysis results including total parameters, trainable parameters,
        model size, and torchinfo summary
    """
    # Set default values
    if device is None:
        device = next(model.parameters()).device

    if dtypes is None:
        # Default to float32 for all inputs, which works for most models
        # Only use int64 if we detect categorical inputs that need it
        dtypes = [torch.float32]

    if col_names is None:
        col_names = [
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
            "trainable",
        ]

    # Get basic model information
    model_info = _get_model_basic_info(model)

    # Print model information
    _print_model_info(model, device, model_info, title, title_color)

    # Generate detailed analysis with torchinfo if input_size is provided
    if input_size is not None:
        input_data = None

        try:
            if isinstance(input_size, tuple):
                # Single input
                input_data = _create_single_input(
                    batch_size, input_size, device, dtypes[0]
                )
            elif isinstance(input_size, list):
                # Multiple inputs - inspect model's forward method
                signature = inspect.signature(model.forward)
                params = list(signature.parameters.values())

                # Skip 'self' parameter
                if params and params[0].name == "self":
                    params = params[1:]

                param_names = [p.name for p in params]

                # Handle multiple inputs based on parameter types
                if len(params) >= 2 and len(input_size) >= 2:
                    # Use more sophisticated input creation for multiple inputs
                    input_data = _create_multi_input_tensors(
                        model, batch_size, input_size, param_names, device, dtypes
                    )
                else:
                    # Generic multiple inputs with simpler approach
                    input_data = _create_generic_inputs(
                        batch_size, input_size, device, dtypes
                    )
        except Exception as e:
            # Fallback to generic handling if inspection fails
            logger.warning(f"Exception during model inspection: {str(e)}")
            if isinstance(input_size, list):
                input_data = _create_generic_inputs(
                    batch_size, input_size, device, dtypes
                )

        # Generate summary with torchinfo
        model_summary = summary(
            model,
            input_data=input_data,
            col_names=col_names,
            verbose=verbose,
            device=device,
        )

        model_info["summary"] = model_summary
    elif verbose > 1:
        # Print basic model structure if no input_size but verbose > 1
        print(f"\n{Fore.YELLOW}Basic Model Structure:{Style.RESET_ALL}")
        print(model)

    return model_info


def _detect_input_shape(model: nn.Module) -> Union[Tuple, List[Tuple]]:
    """Auto-detect appropriate input shape based on model structure."""
    try:
        # Inspect model's forward method
        signature = inspect.signature(model.forward)
        params = list(signature.parameters.values())

        # Skip 'self' parameter
        if params and params[0].name == "self":
            params = params[1:]

        if len(params) == 1:
            # Single input model
            return _detect_single_input_shape(model)
        elif len(params) >= 2:
            # Multi-input model (2 or more inputs)
            return _detect_multi_input_shape(model, params)
        else:
            # No inputs (unusual case)
            logger.warning("Model has no input parameters, using default input shape")
            return (10,)
    except Exception as e:
        # If inspection fails, use a default shape
        logger.warning(f"Could not inspect model: {str(e)}. Using default input shape.")
        return (10,)


def _detect_single_input_shape(model: nn.Module) -> Tuple:
    """Detect input shape for single-input models."""
    # Try to infer input shape from the first Linear layer
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "in_features"):
            input_shape = (module.in_features,)
            logger.info(f"Auto-detected single input shape: {input_shape}")
            return input_shape

    # Default fallback for single input
    input_shape = (10,)
    logger.info(f"Using default single input shape: {input_shape}")
    return input_shape


def _detect_multi_input_shape(model: nn.Module, params: List) -> List[Tuple]:
    """Detect input shapes for models with multiple inputs."""
    param_names = [p.name for p in params]
    logger.info(f"Detected multi-input model with parameters: {param_names}")

    # Try to infer input shapes from model structure
    input_shapes = []

    # First, try to find Linear layers that might correspond to inputs
    linear_layers = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "in_features"):
            linear_layers.append(module.in_features)

    # If we found some Linear layers, use them as hints for input shapes
    if linear_layers:
        # For the first few inputs, use the linear layer dimensions we found
        for i in range(min(len(params), len(linear_layers))):
            input_shapes.append((linear_layers[i],))

    # If we still don't have enough shapes, look for Embedding layers
    if len(input_shapes) < len(params):
        embedding_dims = []
        for _, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                embedding_dims.append(1)  # Categorical features typically have dim 1

        # Add embedding dimensions for remaining inputs
        for i in range(
            len(input_shapes), min(len(params), len(input_shapes) + len(embedding_dims))
        ):
            input_shapes.append((embedding_dims[i - len(input_shapes)],))

    # If we still don't have enough shapes, use default values with different dimensions
    while len(input_shapes) < len(params):
        # Use different default shapes to avoid dimension errors
        input_shapes.append((10 + len(input_shapes),))

    logger.info(f"Auto-detected input shapes for multi-input model: {input_shapes}")
    return input_shapes


def model_info(
    model: nn.Module,
    input_shape: Optional[Union[Tuple, List[Tuple]]] = None,
    batch_size: int = 1,
    **kwargs,
) -> None:
    """
    Display information about a PyTorch model using torchinfo.

    This is a simplified wrapper around analyze_model that automatically detects
    model input structure and configures appropriate inputs. It works with any PyTorch model,
    regardless of its internal architecture or naming conventions.

    Basic usage:
    model_info(model)  # Just pass the model, everything else is automatic

    Advanced usage:
    model_info(model, input_shape=[(128,), (10,)])  # Specify custom input shapes
    model_info(model, batch_size=64, verbose=2)  # Configure display options

    Args:
        model: The PyTorch model to analyze
        input_shape: Optional shape of input tensor(s) to the model (excluding batch dimension)
                    If not provided, will be auto-detected based on model structure
        batch_size: Batch size for the input tensor(s)
        **kwargs: Additional arguments to pass to analyze_model (verbose, title, etc.)
    """
    # Auto-detect input shape if not provided
    if input_shape is None:
        input_shape = _detect_input_shape(model)

    # Set a default title if not provided
    if "title" not in kwargs:
        model_name = model.__class__.__name__
        if hasattr(model, "name"):
            model_name = model.name
        elif hasattr(model, "Name"):
            model_name = model.Name
        kwargs["title"] = f"{model_name} Model Analysis"

    # Call the analyze_model function
    analyze_model(model, input_size=input_shape, batch_size=batch_size, **kwargs)
