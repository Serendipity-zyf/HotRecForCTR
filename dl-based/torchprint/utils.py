"""
Utility functions for model analysis.
"""

import torch
import torch.nn as nn

from typing import (
    Dict,
    List,
    Tuple,
    Union,
)


def count_parameters(module: nn.Module) -> Tuple[int, int]:
    """
    Count the number of parameters in a module.

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = 0
    trainable_params = 0

    for param in module.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    return total_params, trainable_params


def estimate_memory_usage(module: nn.Module, input_size: Tuple[int, ...], output_size: Tuple[int, ...]) -> int:
    """
    Estimate the memory usage of a module in bytes.

    Args:
        module: The module to analyze
        input_size: The input tensor size
        output_size: The output tensor size

    Returns:
        Estimated memory usage in bytes
    """
    # Calculate parameter memory
    param_bytes = 0
    for param in module.parameters():
        param_bytes += param.nelement() * param.element_size()

    # Calculate buffer memory
    buffer_bytes = 0
    for buffer in module.buffers():
        buffer_bytes += buffer.nelement() * buffer.element_size()

    # Estimate input and output memory
    input_bytes = 0
    if input_size:
        input_elements = 1
        for dim in input_size:
            input_elements *= dim
        input_bytes = input_elements * 4  # Assuming float32

    output_bytes = 0
    if output_size:
        output_elements = 1
        for dim in output_size:
            output_elements *= dim
        output_bytes = output_elements * 4  # Assuming float32

    # Total memory
    total_bytes = param_bytes + buffer_bytes + input_bytes + output_bytes

    return total_bytes


def calculate_macs(module: nn.Module, input_size: Tuple[int, ...], output_size: Tuple[int, ...]) -> int:
    """
    Calculate the number of multiply-accumulate operations (MACs) for a module.

    Args:
        module: The module to analyze
        input_size: The input tensor size
        output_size: The output tensor size

    Returns:
        Estimated number of MACs
    """
    macs = 0

    # Handle specific layer types
    if isinstance(module, nn.Linear):
        # For Linear: in_features * out_features * batch_size
        if len(input_size) >= 2:
            batch_size = input_size[0]
            macs = batch_size * module.in_features * module.out_features

    elif isinstance(module, nn.Conv2d):
        # For Conv2d: batch_size * out_channels * out_height * out_width * kernel_height * kernel_width * in_channels / groups
        if len(input_size) == 4 and len(output_size) == 4:  # [batch, channels, height, width]
            batch_size = input_size[0]
            out_h = output_size[2]
            out_w = output_size[3]
            kernel_h, kernel_w = module.kernel_size
            in_channels = module.in_channels // module.groups
            macs = batch_size * module.out_channels * out_h * out_w * kernel_h * kernel_w * in_channels

    elif isinstance(module, nn.Embedding):
        # For Embedding: batch_size * sequence_length * embedding_dim
        if len(input_size) >= 2:
            batch_size = input_size[0]
            seq_len = 1
            for dim in input_size[1:]:
                seq_len *= dim
            macs = batch_size * seq_len * module.embedding_dim

    elif isinstance(module, nn.ModuleList):
        # For ModuleList, we don't calculate MACs directly
        # They will be calculated for each submodule
        pass

    elif isinstance(module, nn.Sequential):
        # For Sequential, we don't calculate MACs directly
        # They will be calculated for each submodule
        pass

    return macs


def calculate_flops(macs: int) -> int:
    """
    Calculate the number of floating point operations (FLOPs) from MACs.

    In neural networks, each multiply-accumulate operation (MAC) typically consists of
    one multiplication and one addition, which equals to 2 floating point operations.

    Args:
        macs: Number of multiply-accumulate operations

    Returns:
        Number of floating point operations (FLOPs)
    """
    # Each MAC operation consists of one multiplication and one addition
    return macs * 2


def get_input_output_sizes(
    model: nn.Module,
    input_data: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
    device: torch.device,
) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    """
    Get input and output sizes for each module in the model.

    Args:
        model: The model to analyze
        input_data: Input data for the model
        device: Device to run the model on

    Returns:
        Dictionary mapping module names to their input and output sizes
    """
    sizes = {}
    hooks = []

    def hook_fn(name):
        def _hook(module, input_tensor, output_tensor):
            # Handle input tensor
            if input_tensor and isinstance(input_tensor, tuple) and len(input_tensor) > 0:
                if isinstance(input_tensor[0], torch.Tensor):
                    input_size = tuple(input_tensor[0].shape)
                else:
                    input_size = None
            else:
                input_size = None

            # Handle output tensor
            if isinstance(output_tensor, torch.Tensor):
                output_size = tuple(output_tensor.shape)
            elif isinstance(output_tensor, (tuple, list)) and output_tensor and isinstance(output_tensor[0], torch.Tensor):
                output_size = tuple(output_tensor[0].shape)
            else:
                output_size = None

            sizes[name] = {"input_size": input_size, "output_size": output_size}

        return _hook

    # Register hooks for all modules
    for name, module in model.named_modules():
        if name == "":  # Skip the root module
            continue
        hooks.append(module.register_forward_hook(hook_fn(name)))

    # Run a forward pass
    with torch.no_grad():
        if isinstance(input_data, (tuple, list)):
            # Move all tensors to the device
            input_data = [x.to(device) if isinstance(x, torch.Tensor) else x for x in input_data]
            model(*input_data)
        else:
            model(input_data.to(device))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return sizes
