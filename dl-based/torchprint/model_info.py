"""
Model analysis functionality for PyTorch models.
"""

import torch
import torch.nn as nn

from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Union,
)

from .utils import (
    count_parameters,
    estimate_memory_usage,
    calculate_macs,
    calculate_flops,
    get_input_output_sizes,
)

from .formatter import (
    format_size,
    format_bytes,
    format_flops,
    format_param_count,
    format_layer_name,
    create_header,
    create_separator,
    colorize_trainable,
    format_with_color,
    strip_ansi,
    TITLE_COLOR,
    HEADER_COLOR,
    SUMMARY_COLOR,
    MEMORY_COLOR,
    SHAPE_COLOR,
    PARAM_COLOR,
    MACS_COLOR,
    FLOPS_COLOR,
    LAYER_NAME_COLOR,
    Style,
)


class ModelAnalyzer:
    """
    Analyzer for PyTorch models that provides detailed information about model structure,
    parameters, memory usage, and computational complexity.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        input_dtypes: Optional[List[torch.dtype]] = None,
    ):
        """
        Initialize the model analyzer.

        Args:
            model: The PyTorch model to analyze
            model_name: Optional name for the model (defaults to model.__class__.__name__)
            device: Device to run the analysis on (defaults to CUDA if available, else CPU)
            input_dtypes: List of dtypes for input tensors (defaults to torch.float32)
        """
        self.model = model
        self.model_name = model_name or model.__class__.__name__
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.input_dtypes = input_dtypes or [torch.float32]

        # Move model to the specified device
        self.model.to(self.device)

        # Initialize analysis results
        self.total_params = 0
        self.trainable_params = 0
        self.total_macs = 0
        self.total_flops = 0
        self.total_memory = 0
        self.module_info = {}
        self.input_output_sizes = {}

        # For tree structure tracking
        self.module_depth_idx = {}
        self.depth_counts = {}

    def analyze(
        self,
        input_data: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]
        ] = None,
        input_size: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
        input_dims: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        long_indices: Optional[List[int]] = None,
        batch_size: int = 1,
    ) -> None:
        """
        Analyze the model structure, parameters, and computational complexity.

        Args:
            input_data: Input data for the model. If provided, a forward pass will be performed.
            input_size: Input size for the model. If provided and input_data is None,
                        random tensors of this size will be created.
            input_dims: Input dimensions excluding batch dimension. If provided and input_size is None,
                        random tensors will be created with the specified batch size and these dimensions.
            dtypes: Data types for the input tensors. Defaults to self.input_dtypes.
            long_indices: List of indices for inputs that should have torch.long dtype.
                          For example, [1] means the second input should be torch.long.
            batch_size: Batch size to use when creating random tensors. Defaults to 1.
        """
        # Prepare input data if not provided
        if input_data is None:
            if input_size is not None:
                # Create random tensors with the specified sizes
                if isinstance(input_size, tuple) and all(isinstance(x, int) for x in input_size):
                    # Single input tensor
                    input_data = torch.rand(
                        input_size, dtype=dtypes[0] if dtypes else self.input_dtypes[0]
                    )
                else:
                    # Multiple input tensors
                    input_data = [
                        torch.rand(
                            size,
                            dtype=(
                                dtypes[i] if dtypes and i < len(dtypes) else self.input_dtypes[0]
                            ),
                        )
                        for i, size in enumerate(input_size)
                    ]
            elif input_dims is not None:
                # Create random tensors with specified batch size and the specified dimensions
                if isinstance(input_dims, tuple) and all(isinstance(x, int) for x in input_dims):
                    # Single input tensor
                    dtype = dtypes[0] if dtypes else self.input_dtypes[0]
                    # Override dtype if this is a long index
                    if long_indices and 0 in long_indices:
                        dtype = torch.long
                    # Create tensor with appropriate batch size and dimensions
                    if dtype == torch.long:
                        # For long dtype, create integer tensor directly
                        input_data = torch.randint(0, 2, (batch_size,) + input_dims)
                    else:
                        input_data = torch.rand(
                            (batch_size,) + input_dims,
                            dtype=dtype,
                        )
                else:
                    # Multiple input tensors
                    input_data = []
                    for i, dims in enumerate(input_dims):
                        # Determine dtype for this tensor
                        if dtypes and i < len(dtypes):
                            dtype = dtypes[i]
                        else:
                            dtype = self.input_dtypes[0]

                        # Override dtype if this is a long index
                        if long_indices and i in long_indices:
                            dtype = torch.long

                        # Create tensor with appropriate batch size and dimensions
                        if dtype == torch.long:
                            # For long dtype, create integer tensor directly
                            tensor = torch.randint(0, 10, (batch_size,) + dims)
                        else:
                            tensor = torch.rand((batch_size,) + dims, dtype=dtype)

                        input_data.append(tensor)

        # Get input and output sizes for each module
        if input_data is not None:
            self.input_output_sizes = get_input_output_sizes(self.model, input_data, self.device)

        # Count parameters
        self.total_params, self.trainable_params = count_parameters(self.model)

        # Assign depth-idx to each module
        self._assign_depth_idx()

        # Analyze each module
        for name, module in self.model.named_modules():
            if name == "":  # Skip the root module
                continue

            # Get parameter counts for this module
            params, trainable_params = count_parameters(module)

            # Get input and output sizes
            input_size = None
            output_size = None
            if name in self.input_output_sizes:
                input_size = self.input_output_sizes[name]["input_size"]
                output_size = self.input_output_sizes[name]["output_size"]

            # Calculate MACs and FLOPs
            macs = 0
            flops = 0
            if input_size and output_size:
                macs = calculate_macs(module, input_size, output_size)
                flops = calculate_flops(macs)
                self.total_macs += macs
                self.total_flops += flops

            # Estimate memory usage
            memory = estimate_memory_usage(module, input_size, output_size)
            self.total_memory += memory

            # Store module info
            self.module_info[name] = {
                "module": module,
                "type": module.__class__.__name__,
                "params": params,
                "trainable_params": trainable_params,
                "trainable": any(p.requires_grad for p in module.parameters()),
                "input_size": input_size,
                "output_size": output_size,
                "macs": macs,
                "flops": flops,
                "memory": memory,
                "depth_idx": self.module_depth_idx.get(name, ""),
            }

    def _assign_depth_idx(self) -> None:
        """Assign depth-idx identifiers to each module."""
        # Reset counters
        self.depth_counts = {}

        # Process modules in order
        for name, module in self.model.named_modules():
            if name == "":  # Skip the root module
                continue

            # Calculate depth
            depth = len(name.split("."))

            # Increment counter for this depth
            if depth not in self.depth_counts:
                self.depth_counts[depth] = 0
            self.depth_counts[depth] += 1

            # Assign depth-idx
            self.module_depth_idx[name] = f"{depth}-{self.depth_counts[depth]}"

    def _get_module_depth(self, name: str) -> int:
        """Get the depth of a module based on its name."""
        return len(name.split("."))

    def _is_leaf_module(self, name: str) -> bool:
        """Check if a module is a leaf module (has no child modules)."""
        prefix = name + "."
        return not any(other_name.startswith(prefix) for other_name in self.module_info.keys())

    def _calculate_max_name_length(self, max_depth: int) -> int:
        """Calculate the maximum length of formatted module names."""
        max_length = len(self.model_name)  # Start with root module name length

        # Helper function to calculate formatted name length
        def get_formatted_name_length(name: str, depth: int) -> int:
            if depth == 0:
                return len(name)

            # Get module info
            info = self.module_info.get(name, {})
            module_type = info.get("type", "")
            depth_idx = info.get("depth_idx", "")

            # Get the short name (last part of the full name)
            short_name = name.split(".")[-1]

            # Calculate tree prefix length (approximately)
            prefix_length = depth * 4  # Each level adds about 4 characters

            # Calculate total visible length
            if depth_idx:
                return (
                    prefix_length + len(short_name) + len(module_type) + len(depth_idx) + 4
                )  # 4 for separators
            else:
                return prefix_length + len(short_name) + len(module_type) + 2  # 2 for separator

        # Check all modules
        for name in self.module_info.keys():
            depth = len(name.split("."))
            if depth <= max_depth:
                name_length = get_formatted_name_length(name, depth)
                max_length = max(max_length, name_length)

        return max_length

    def _get_parent_name(self, name: str) -> str:
        """Get the parent module name."""
        parts = name.split(".")
        if len(parts) == 1:
            return ""
        return ".".join(parts[:-1])

    def _get_child_modules(self, name: str) -> List[str]:
        """Get the direct child modules of a module in their registration order."""
        if name == "":
            # For the root module, get the top-level modules in their registration order
            module = self.model
            return [child_name for child_name in module._modules.keys()]
        else:
            # For non-root modules, find the module and get its children
            module = self.model
            for part in name.split("."):
                if part in module._modules:
                    module = module._modules[part]
                else:
                    # Module not found, fall back to the old method
                    prefix = name + "." if name else ""
                    children = []
                    for other_name in self.module_info.keys():
                        if other_name.startswith(prefix) and other_name != name:
                            # Check if it's a direct child
                            remaining = other_name[len(prefix) :]
                            if "." not in remaining:
                                children.append(other_name)
                    return sorted(children)

            # Get the children of this module in their registration order
            return [f"{name}.{child_name}" for child_name in module._modules.keys()]

    def _format_module_tree(self, name: str = "", depth: int = 0, max_depth: int = 3) -> List[str]:
        """Format the module tree for display."""
        lines = []
        prefix_dict = {}  # Track vertical lines for tree structure

        # First pass: calculate the maximum module name length
        max_name_length = self._calculate_max_name_length(max_depth)
        # Ensure minimum width of 40 characters
        col_width_name = max(40, max_name_length + 5)  # Add some padding

        # Adjust column widths
        col_widths = [col_width_name, 20, 20, 12, 12, 10]

        if depth > max_depth:
            return lines

        if name == "":
            # Root module
            module_type = self.model.__class__.__name__

            # Find the input and output sizes for the root module
            root_input_size = None
            root_output_size = None

            # Try to find the first layer's input and the last layer's output
            if self.input_output_sizes:
                # For input, find the first layer's input
                first_layers = self._get_child_modules("")
                if first_layers:
                    first_layer = first_layers[0]
                    if first_layer in self.input_output_sizes:
                        root_input_size = self.input_output_sizes[first_layer]["input_size"]

                # For output, use the model's direct output if available
                # This is a simplification and might not be accurate for all models
                all_modules = list(self.input_output_sizes.keys())
                if all_modules:
                    last_module = all_modules[-1]
                    root_output_size = self.input_output_sizes[last_module]["output_size"]

            # Format each column with adjusted width and color
            col1 = format_with_color(self.model_name, col_widths[0], LAYER_NAME_COLOR)
            col2 = format_with_color(format_size(root_input_size), col_widths[1], SHAPE_COLOR)
            col3 = format_with_color(format_size(root_output_size), col_widths[2], SHAPE_COLOR)
            col4 = format_with_color(
                format_param_count(self.total_params), col_widths[3], PARAM_COLOR
            )
            col5 = format_with_color(format_flops(self.total_macs), col_widths[4], MACS_COLOR)
            col6 = colorize_trainable(True)

            # Construct the line
            line = f"{col1}{col2}{col3}{col4}{col5}{col6}"
            lines.append(line)

            # Add child modules in their registration order
            children = self._get_child_modules("")

            # Convert child names to full module names
            full_children = []
            for child_name in children:
                # Check if the child name exists in module_info
                if child_name in self.module_info:
                    full_children.append(child_name)
                else:
                    # Try to find it with the full path
                    for module_name in self.module_info.keys():
                        if module_name.endswith("." + child_name) or module_name == child_name:
                            full_children.append(module_name)
                            break

            for i, child in enumerate(full_children):
                is_last = i == len(full_children) - 1
                child_lines = self._format_module_tree_recursive(
                    child, depth + 1, max_depth, is_last, prefix_dict.copy(), col_widths
                )
                lines.extend(child_lines)

            return lines

        return self._format_module_tree_recursive(
            name, depth, max_depth, False, prefix_dict, col_widths
        )

    def _format_module_tree_recursive(
        self,
        name: str,
        depth: int,
        max_depth: int,
        is_last: bool,
        prefix_dict: Dict[int, bool],
        col_widths: List[int] = None,
    ) -> List[str]:
        """Recursively format the module tree for display."""
        if depth > max_depth:
            return []

        lines = []

        # Get module info
        info = self.module_info[name]
        module_type = info["type"]
        params = info["params"]
        trainable = info["trainable"]
        input_size = info["input_size"]
        output_size = info["output_size"]
        macs = info["macs"]
        depth_idx = info["depth_idx"]

        # Get the short name (last part of the full name)
        short_name = name.split(".")[-1]

        # Format the line with tree structure
        module_name = format_layer_name(
            short_name, module_type, depth, is_last, prefix_dict, depth_idx
        )

        # Use provided column widths or default
        if col_widths is None:
            col_widths = [40, 20, 20, 12, 12, 10]  # Default widths

        # For the first column (module name), we don't use format_with_color because it already has colors
        # and tree structure. Instead, we'll calculate the padding manually.
        visible_length = len(strip_ansi(module_name))
        padding = max(0, col_widths[0] - visible_length)
        col1 = module_name + " " * padding

        col2 = format_with_color(format_size(input_size), col_widths[1], SHAPE_COLOR)
        col3 = format_with_color(format_size(output_size), col_widths[2], SHAPE_COLOR)
        col4 = format_with_color(format_param_count(params), col_widths[3], PARAM_COLOR)
        col5 = format_with_color(format_flops(macs), col_widths[4], MACS_COLOR)
        col6 = colorize_trainable(trainable)

        # Construct the line
        line = f"{col1}{col2}{col3}{col4}{col5}{col6}"
        lines.append(line)

        # Add child modules
        children = self._get_child_modules(name)
        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            child_lines = self._format_module_tree_recursive(
                child, depth + 1, max_depth, child_is_last, prefix_dict.copy(), col_widths
            )
            lines.extend(child_lines)

        return lines

    def summary(self, max_depth: int = 3) -> str:
        """
        Generate a summary of the model.

        Args:
            max_depth: Maximum depth of modules to display

        Returns:
            Formatted summary string
        """
        lines = []

        # First pass: calculate the maximum module name length
        max_name_length = self._calculate_max_name_length(max_depth)
        # Ensure minimum width of 40 characters
        col_width_name = max(40, max_name_length + 5)  # Add some padding

        # Adjust column widths
        col_widths = [col_width_name, 20, 20, 12, 12, 10]

        # Add title
        lines.append(f"{TITLE_COLOR}{self.model_name} Model Analysis{Style.RESET_ALL}")
        lines.append(create_separator(col_widths))

        # Add model info
        lines.append(f"{HEADER_COLOR}Model Name:{Style.RESET_ALL} {self.model_name}")
        lines.append(f"{HEADER_COLOR}Analysis device:{Style.RESET_ALL} {self.device}")
        lines.append(create_separator(col_widths))

        # Add header
        lines.append(create_header(col_widths))
        lines.append(create_separator(col_widths))

        # Add module tree
        tree_lines = self._format_module_tree(max_depth=max_depth)
        lines.extend(tree_lines)

        # Add summary
        lines.append(create_separator(col_widths))
        lines.append(
            f"{SUMMARY_COLOR}Total params:{Style.RESET_ALL} {PARAM_COLOR}{self.total_params:,}{Style.RESET_ALL}"
        )
        lines.append(
            f"{SUMMARY_COLOR}Trainable params:{Style.RESET_ALL} {PARAM_COLOR}{self.trainable_params:,}{Style.RESET_ALL}"
        )
        lines.append(
            f"{SUMMARY_COLOR}Non-trainable params:{Style.RESET_ALL} {PARAM_COLOR}{self.total_params - self.trainable_params:,}{Style.RESET_ALL}"
        )
        lines.append(
            f"{SUMMARY_COLOR}Total mult-adds:{Style.RESET_ALL} {MACS_COLOR}{format_flops(self.total_macs)}{Style.RESET_ALL}"
        )
        lines.append(
            f"{SUMMARY_COLOR}Total FLOPs:{Style.RESET_ALL} {FLOPS_COLOR}{format_flops(self.total_flops)}{Style.RESET_ALL}"
        )
        lines.append(create_separator(col_widths))

        # Add memory usage
        input_size = 0  # Placeholder for input size
        forward_backward_size = 0  # Placeholder for forward/backward pass size
        params_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        total_size = params_size + input_size + forward_backward_size

        lines.append(f"{MEMORY_COLOR}Input size:{Style.RESET_ALL} {format_bytes(input_size)}")
        lines.append(
            f"{MEMORY_COLOR}Forward/backward pass size:{Style.RESET_ALL} {format_bytes(forward_backward_size)}"
        )
        lines.append(f"{MEMORY_COLOR}Params size:{Style.RESET_ALL} {format_bytes(params_size)}")
        lines.append(
            f"{MEMORY_COLOR}Estimated Total Size:{Style.RESET_ALL} {format_bytes(total_size)}"
        )

        return "\n".join(lines)

    def print_summary(self, max_depth: int = 3) -> None:
        """Print the model summary."""
        print(self.summary(max_depth))


def analyze_model(
    model: nn.Module,
    model_name: Optional[str] = None,
    input_data: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]] = None,
    input_size: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    input_dims: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    device: Optional[torch.device] = None,
    dtypes: Optional[List[torch.dtype]] = None,
    long_indices: Optional[List[int]] = None,
    batch_size: int = 1,
    max_depth: int = 3,
) -> str:
    """
    Analyze a PyTorch model and return a formatted summary.

    Args:
        model: The PyTorch model to analyze
        model_name: Optional name for the model (defaults to model.__class__.__name__)
        input_data: Input data for the model. If provided, a forward pass will be performed.
        input_size: Input size for the model. If provided and input_data is None,
                    random tensors of this size will be created.
        input_dims: Input dimensions excluding batch dimension. If provided and input_size is None,
                    random tensors will be created with batch size and these dimensions.
        device: Device to run the analysis on (defaults to CUDA if available, else CPU)
        dtypes: Data types for the input tensors. Defaults to torch.float32.
        long_indices: List of indices for inputs that should have torch.long dtype.
                      For example, [1] means the second input should be torch.long.
        batch_size: Batch size to use when creating random tensors. Defaults to 1.
        max_depth: Maximum depth of modules to display

    Returns:
        Formatted summary string
    """
    analyzer = ModelAnalyzer(model, model_name, device, dtypes)
    analyzer.analyze(input_data, input_size, input_dims, dtypes, long_indices, batch_size)
    return analyzer.summary(max_depth)
