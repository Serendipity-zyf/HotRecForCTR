"""
Complete demo script for the enhanced torchprint module.

This script demonstrates all the features of the enhanced torchprint module:
1. FLOPS calculation with automatic unit selection (K, M, G)
2. Input dimensions specification without batch dimension
3. Long dtype specification for specific inputs
4. Analysis of various model architectures (CNN, Transformer, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import ColorLogger
from torchprint import analyze_model
from models.FMCTR import FMCTR

logger = ColorLogger(name="EnhancedDemo")


def create_fmctr_model():
    """Create an FMCTR model for demonstration."""
    feature_dims = [10, 20, 30, 40, 50]
    dense_feature_dim = 13
    embed_dim = 8
    return FMCTR(feature_dims, dense_feature_dim, embed_dim)


def create_cnn_model():
    """Create a CNN model for demonstration."""
    
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            
            # Pooling layer
            self.pool = nn.MaxPool2d(2, 2)
            
            # Fully connected layers
            self.fc1 = nn.Linear(64 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)
            
            # Dropout
            self.dropout = nn.Dropout(0.25)
        
        def forward(self, x):
            # Convolutional layers with ReLU and pooling
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            
            # Flatten
            x = x.view(-1, 64 * 4 * 4)
            
            # Fully connected layers with ReLU and dropout
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            
            return x
    
    return CNN()


def create_transformer_model():
    """Create a Transformer model for demonstration."""
    
    class TransformerModel(nn.Module):
        def __init__(self, vocab_size=10000, embed_dim=512, num_heads=8, 
                     hidden_dim=2048, num_layers=6, max_seq_length=100):
            super().__init__()
            
            # Embedding layers
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.position_embedding = nn.Parameter(torch.zeros(max_seq_length, embed_dim))
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Output layer
            self.output_layer = nn.Linear(embed_dim, vocab_size)
            
            # Initialize parameters
            self.init_weights()
        
        def init_weights(self):
            initrange = 0.1
            self.token_embedding.weight.data.uniform_(-initrange, initrange)
            self.output_layer.bias.data.zero_()
            self.output_layer.weight.data.uniform_(-initrange, initrange)
        
        def forward(self, x):
            # Get sequence length
            seq_length = x.size(1)
            
            # Create embeddings
            token_embeddings = self.token_embedding(x)
            position_embeddings = self.position_embedding[:seq_length, :].unsqueeze(0).expand(x.size(0), -1, -1)
            embeddings = token_embeddings + position_embeddings
            
            # Pass through transformer
            transformer_output = self.transformer_encoder(embeddings)
            
            # Get output
            output = self.output_layer(transformer_output)
            
            return output
    
    return TransformerModel()


def create_lstm_model():
    """Create an LSTM model for demonstration."""
    
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512, 
                     num_layers=2, dropout=0.5):
            super().__init__()
            
            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            
            # Output layer
            self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # *2 for bidirectional
            
            # Dropout
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            # Embedding
            embedded = self.embedding(x)
            
            # LSTM
            lstm_output, _ = self.lstm(embedded)
            
            # Get the output from the last time step
            lstm_output = self.dropout(lstm_output)
            
            # Pass through linear layer
            output = self.fc(lstm_output)
            
            return output
    
    return LSTMModel()


def demo_enhanced():
    """Demonstrate all features of the enhanced torchprint module with various models."""
    logger.info("Starting enhanced demo with multiple model architectures")

    print("\n" + "=" * 70)
    print("ENHANCED TORCHPRINT MODULE DEMO WITH MULTIPLE ARCHITECTURES")
    print("=" * 70)

    # Create and analyze FMCTR model
    print("\n" + "=" * 50)
    print("1. ANALYZING FMCTR MODEL (RECOMMENDATION)")
    print("=" * 50)
    fmctr_model = create_fmctr_model()
    summary = analyze_model(
        fmctr_model,
        model_name="FMCTR",
        input_dims=[(13,), (5,)],  # Two inputs with dimensions (13,) and (5,)
        long_indices=[1],          # Second input (index 1) should be torch.long
        batch_size=128,            # Specify batch size
    )
    print(summary)

    # Create and analyze CNN model
    print("\n" + "=" * 50)
    print("2. ANALYZING CNN MODEL (COMPUTER VISION)")
    print("=" * 50)
    cnn_model = create_cnn_model()
    summary = analyze_model(
        cnn_model,
        model_name="CNN",
        input_dims=(3, 32, 32),    # Input dimensions for image (channels, height, width)
        batch_size=64,             # Specify batch size
    )
    print(summary)

    # Create and analyze Transformer model
    print("\n" + "=" * 50)
    print("3. ANALYZING TRANSFORMER MODEL (NLP)")
    print("=" * 50)
    transformer_model = create_transformer_model()
    summary = analyze_model(
        transformer_model,
        model_name="Transformer",
        input_dims=(50,),          # Sequence length
        long_indices=[0],          # Input should be torch.long (token IDs)
        batch_size=32,             # Specify batch size
    )
    print(summary)

    # Create and analyze LSTM model
    print("\n" + "=" * 50)
    print("4. ANALYZING LSTM MODEL (NLP)")
    print("=" * 50)
    lstm_model = create_lstm_model()
    summary = analyze_model(
        lstm_model,
        model_name="LSTM",
        input_dims=(100,),         # Sequence length
        long_indices=[0],          # Input should be torch.long (token IDs)
        batch_size=16,             # Specify batch size
    )
    print(summary)

    print("\n" + "=" * 70)
    print("DEMO COMPLETED!")
    print("=" * 70)
    print("\nEnhanced features demonstrated:")
    print("1. FLOPS calculation with automatic unit selection (K, M, G)")
    print("2. Input dimensions without batch dimension specification")
    print("3. Long dtype specification for token inputs")
    print("4. Analysis of various model architectures (CNN, Transformer, LSTM)")


if __name__ == "__main__":
    demo_enhanced()
