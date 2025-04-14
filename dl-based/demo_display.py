"""
Demo script to showcase the dictionary display functionality.
"""

from utils.display import display_dict
from utils.display import display_nested_dict
from utils.display import pretty_dict
from utils.logger import ColorLogger

logger = ColorLogger(name="DisplayDemo")


def demo_display():
    """Demonstrate the dictionary display functionality."""
    logger.info("Starting display demo")

    print("\n" + "=" * 50)
    print("DICTIONARY DISPLAY DEMO")
    print("=" * 50)

    # Simple dictionary with various value types
    simple_dict = {
        "name": "FM Model",
        "learning_rate": 0.01,
        "batch_size": 256,
        "epochs": 10,
        "optimizer_params": (0.9, 0.999),
        "layers": [64, 32, 16],
        "use_bias": True,
        "dropout": 0.5,
    }

    # Display the simple dictionary
    display_dict(simple_dict, title="Model Configuration")

    # Dictionary with nested structure
    nested_dict = {
        "model": {
            "name": "DeepFM",
            "embedding_size": 16,
            "hidden_units": [64, 32],
            "dropout": 0.2,
        },
        "training": {
            "batch_size": 256,
            "epochs": 20,
            "learning_rate": 0.001,
            "optimizer": {
                "name": "Adam",
                "params": {
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                },
            },
        },
        "evaluation": {
            "metrics": ["auc", "logloss", "accuracy"],
            "validation_split": 0.2,
        },
        "data": {
            "features": ["feature1", "feature2", "feature3", "feature4", "feature5"],
            "target": "click",
            "categorical_cols": ["feature1", "feature3"],
            "numerical_cols": ["feature2", "feature4", "feature5"],
        },
    }

    # Display the nested dictionary in table format
    display_dict(nested_dict, title="Complete Configuration (Table Format)")

    # Display the nested dictionary in nested format
    display_nested_dict(nested_dict, title="Complete Configuration (Nested Format)")

    # Use the convenience function
    print("\nUsing pretty_dict with table_format=True:")
    pretty_dict(simple_dict, title="Model Configuration", table_format=True)

    print("\nUsing pretty_dict with table_format=False:")
    pretty_dict(nested_dict, title="Complete Configuration", table_format=False)

    print("\nDemo completed!")


if __name__ == "__main__":
    demo_display()
