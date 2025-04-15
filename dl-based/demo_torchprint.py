"""
Complete demo script for the enhanced torchprint module.

This script demonstrates all the features of the enhanced torchprint module:
1. FLOPS calculation
2. Input dimensions specification without batch dimension
3. Long dtype specification for specific inputs
"""

from utils.logger import ColorLogger
from torchprint import analyze_model
from models.FMCTR import FMCTR

logger = ColorLogger(name="CompleteDemo")


def create_fmctr_model():
    """Create an FMCTR model for demonstration."""
    feature_dims = [10, 20, 30, 40, 50]
    dense_feature_dim = 13
    embed_dim = 8
    return FMCTR(feature_dims, dense_feature_dim, embed_dim)


def demo_complete():
    """Demonstrate all features of the enhanced torchprint module."""
    logger.info("Starting complete demo for enhanced torchprint module")

    print("\n" + "=" * 50)
    print("ENHANCED TORCHPRINT MODULE DEMO")
    print("=" * 50)

    # Create FMCTR model
    fmctr_model = create_fmctr_model()

    # Analyze model with input dimensions and long indices
    print("\nAnalyzing FMCTR model with enhanced features:")
    summary = analyze_model(
        fmctr_model,
        model_name="FMCTR",
        input_dims=[(13,), (5,)],  # Two inputs with dimensions (13,) and (5,)
        long_indices=[1],  # Second input (index 1) should be torch.long
        batch_size=128,  # Specify batch size
    )
    print(summary)

    print("\nDemo completed!")
    print("\nEnhanced features:")
    print("1. FLOPS calculation - Shows computational complexity (2x MACs)")
    print("2. Input dimensions without batch dimension - Easier to specify model inputs")
    print("3. Long dtype specification - Specify which inputs should be integers")


if __name__ == "__main__":
    demo_complete()
