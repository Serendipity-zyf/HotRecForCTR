"""
Test script for the refactored model_utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from utils.model_utils import model_info


class SingleInputModel(nn.Module):
    """A simple model with a single input."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        self.name = "SingleInputModel"
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class DualInputModel(nn.Module):
    """A model with two inputs: continuous and categorical."""

    def __init__(
        self, cont_dim=10, cat_dims=[5, 5, 5], emb_dim=4, hidden_dim=20, output_dim=1
    ):
        super().__init__()
        self.name = "DualInputModel"

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, emb_dim) for dim in cat_dims]
        )

        # Layer for continuous features
        self.cont_layer = nn.Linear(cont_dim, hidden_dim)

        # Combined layers
        total_emb_dim = len(cat_dims) * emb_dim
        self.combined_layer = nn.Linear(hidden_dim + total_emb_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, cont_x, cat_x):
        # Process continuous features
        cont_out = torch.relu(self.cont_layer(cont_x))

        # Process categorical features
        embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_out = torch.cat(embs, dim=1)

        # Combine and output
        combined = torch.cat([cont_out, cat_out], dim=1)
        hidden = torch.relu(self.combined_layer(combined))
        return torch.sigmoid(self.output_layer(hidden))


class MultiInputModel(nn.Module):
    """A model with three different inputs."""

    def __init__(
        self, input1_dim=10, input2_dim=5, input3_dim=3, hidden_dim=20, output_dim=1
    ):
        super().__init__()
        self.name = "MultiInputModel"

        # Layers for each input
        self.layer1 = nn.Linear(input1_dim, hidden_dim)
        self.layer2 = nn.Linear(input2_dim, hidden_dim)
        self.layer3 = nn.Linear(input3_dim, hidden_dim)

        # Combined layer
        self.combined_layer = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1, x2, x3):
        # Process each input
        out1 = torch.relu(self.layer1(x1))
        out2 = torch.relu(self.layer2(x2))
        out3 = torch.relu(self.layer3(x3))

        # Combine and output
        combined = torch.cat([out1, out2, out3], dim=1)
        hidden = torch.relu(self.combined_layer(combined))
        return torch.sigmoid(self.output_layer(hidden))


class FMCTR(nn.Module):
    """Factorization Machine for CTR prediction."""

    def __init__(self, feature_dims: List[int], dense_feature_dim: int, embed_dim: int):
        super(FMCTR, self).__init__()
        self.name = "FMCTR"
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, embed_dim) for dim in feature_dims]
        )
        self.dense_layer = nn.Linear(dense_feature_dim, embed_dim)

    def forward(self, dense_x: torch.Tensor, discrete_x: torch.Tensor) -> torch.Tensor:
        # discrete_embeds shape's [batch_size, num_discrete_features, embed_dim]
        discret_embeds = torch.stack(
            [emb(discrete_x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )
        # dense_embed shape's [batch_size, embed_dim]
        dense_embed = self.dense_layer(dense_x)
        # embeds shape's [batch_size, num_discrete_features + 1, embed_dim]
        embeds = torch.cat([discret_embeds, dense_embed.unsqueeze(1)], dim=1)
        # FM Calculation
        # sum_of_embeds shape's [batch_size, embed_dim]
        sum_of_embeds = torch.sum(embeds, dim=1)
        # square_of_sum shape's [batch_size, embed_dim]
        square_of_sum = torch.square(sum_of_embeds)
        # sum_of_square shape's [batch_size, embed_dim]
        sum_of_square = torch.sum(torch.square(embeds), dim=1)
        # fm_output shape's [batch_size]
        fm_output = torch.sum(0.5 * (square_of_sum - sum_of_square), dim=1)
        return fm_output


class SelfAttentionModel(nn.Module):
    """A model that uses self-attention mechanism."""

    def __init__(self, input_dim=10, hidden_dim=20, num_heads=2, output_dim=1):
        super().__init__()
        self.name = "SelfAttentionModel"

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Project input to hidden dimension
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)

        # Self-attention expects [seq_len, batch_size, hidden_dim]
        x = x.transpose(0, 1)

        # Apply self-attention
        attn_output, _ = self.self_attention(x, x, x)

        # Residual connection and layer normalization
        x = self.layer_norm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x)

        # Residual connection and layer normalization
        x = self.layer_norm2(x + ffn_output)

        # Convert back to [batch_size, seq_len, hidden_dim]
        x = x.transpose(0, 1)

        # Average pooling over sequence dimension
        x = torch.mean(x, dim=1)

        # Output projection
        return torch.sigmoid(self.output_layer(x))


class CNNModel(nn.Module):
    """A convolutional neural network model for image data."""

    def __init__(self, in_channels=3, img_size=32, num_classes=10):
        super().__init__()
        self.name = "CNNModel"

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size after convolutions and pooling
        # After 3 pooling layers of stride 2, the size is reduced by factor of 2^3 = 8
        final_size = img_size // 8

        # Fully connected layers
        self.fc1 = nn.Linear(128 * final_size * final_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input shape: [batch_size, channels, height, width]
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class ComplexMultiInputModel(nn.Module):
    """A complex model with multiple inputs of different types."""

    def __init__(
        self, input1_dim=10, input2_dim=20, input3_dim=30, hidden_dim=64, output_dim=1
    ):
        super().__init__()
        self.name = "ComplexMultiInputModel"

        # Process first input - numerical features
        self.input1_layers = nn.Sequential(
            nn.Linear(input1_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Process second input - categorical features
        self.input2_layers = nn.Sequential(
            nn.Linear(input2_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Process third input - text features
        self.input3_layers = nn.Sequential(
            nn.Linear(input3_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Combine all features
        self.combined_layer = nn.Linear(hidden_dim * 3, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2, x3):
        # Process each input
        out1 = self.input1_layers(x1)
        out2 = self.input2_layers(x2)
        out3 = self.input3_layers(x3)

        # Combine all features
        combined = torch.cat([out1, out2, out3], dim=1)
        combined = F.relu(self.bn(self.combined_layer(combined)))

        # Final output
        return self.output_layers(combined)


def test_model_utils():
    """Test the model_utils functions with different model types."""
    print("\n===== Testing model_utils with different model types =====\n")

    # Test with single input model
    print("\n----- Testing with SingleInputModel -----")
    model1 = SingleInputModel()
    model_info(model1)

    # Test with dual input model
    print("\n----- Testing with DualInputModel -----")
    model2 = DualInputModel()
    model_info(model2)

    # Test with multi-input model
    print("\n----- Testing with MultiInputModel -----")
    model3 = MultiInputModel()
    model_info(model3)

    # Test with FMCTR model
    print("\n----- Testing with FMCTR model -----")
    feature_dims = [10, 20, 30]  # Example feature dimensions
    dense_feature_dim = 5  # Example dense feature dimension
    embed_dim = 8  # Example embedding dimension
    model4 = FMCTR(feature_dims, dense_feature_dim, embed_dim)

    # For FMCTR, we need to specify the input shapes explicitly
    # First shape is for dense_x (dense_feature_dim,)
    # Second shape is for discrete_x (len(feature_dims),)
    # We also need to specify the dtypes to ensure correct tensor types
    model_info(
        model4,
        input_shape=[(dense_feature_dim,), (len(feature_dims),)],
        dtypes=[
            torch.float32,
            torch.long,
        ],  # Ensure discrete_x is created as long tensor
    )

    # Test with attention-based model
    print("\n----- Testing with SelfAttentionModel -----")
    model5 = SelfAttentionModel(input_dim=10, hidden_dim=16, num_heads=2)
    # For attention models, we need to specify the sequence length
    model_info(model5, input_shape=[(5, 10)])  # [seq_len, input_dim]

    # Test with CNN model (C,H,W format)
    print("\n----- Testing with CNNModel (C,H,W format) -----")
    model6 = CNNModel(in_channels=3, img_size=32)
    # For CNN models, we specify channels, height, width (C,H,W)
    model_info(model6, input_shape=(3, 32, 32))

    # Test with complex multi-input model
    print("\n----- Testing with ComplexMultiInputModel -----")
    model7 = ComplexMultiInputModel(input1_dim=10, input2_dim=20, input3_dim=30)
    # For multi-input models with different feature dimensions
    model_info(model7, input_shape=[(10,), (20,), (30,)])

    print("\n===== All tests completed =====")


if __name__ == "__main__":
    test_model_utils()
