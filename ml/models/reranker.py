"""Neural re-ranker for career recommendations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)


class CareerReRanker(nn.Module):
    """
    Multi-layer perceptron for re-ranking career recommendations.
    
    Architecture:
    - Input: User vector (914-dim) + Career vector (515-dim) + Interactions (1429-dim)
    - Hidden layers: 512 → 256 → 128
    - Output: Single score (0-1)
    - Normalization: BatchNorm1d after each hidden layer
    - Regularization: Dropout(0.3)
    - Activation: ReLU
    """
    
    def __init__(
        self,
        user_dim: int = 914,
        career_dim: int = 515,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3
    ):
        """
        Initialize re-ranker model.
        
        Args:
            user_dim: User vector dimensionality
            career_dim: Career vector dimensionality
            hidden_dims: Hidden layer sizes
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.user_dim = user_dim
        self.career_dim = career_dim
        
        # Interaction features: element-wise product + L1 distance
        # Size: user_dim + career_dim + (user_dim + career_dim)
        self.input_dim = user_dim + career_dim + (user_dim + career_dim)
        
        # Build MLP layers
        layers = []
        in_features = self.input_dim
        
        for hidden_size in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def compute_interaction_features(
        self,
        user_vectors: torch.Tensor,
        career_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interaction features between user and career vectors.
        
        Args:
            user_vectors: Batch of user vectors (batch_size, user_dim)
            career_vectors: Batch of career vectors (batch_size, career_dim)
        
        Returns:
            Concatenated features with interactions
        """
        # Align dimensions (pad shorter vector with zeros if needed)
        max_dim = max(user_vectors.size(1), career_vectors.size(1))
        
        user_padded = torch.nn.functional.pad(
            user_vectors,
            (0, max_dim - user_vectors.size(1))
        )
        career_padded = torch.nn.functional.pad(
            career_vectors,
            (0, max_dim - career_vectors.size(1))
        )
        
        # Element-wise product (captures feature alignment)
        product = user_padded * career_padded
        
        # L1 distance (captures feature difference)
        l1_dist = torch.abs(user_padded - career_padded)
        
        # Concatenate all features
        combined = torch.cat([
            user_vectors,
            career_vectors,
            product,
            l1_dist
        ], dim=1)
        
        return combined
    
    def forward(
        self,
        user_vectors: torch.Tensor,
        career_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_vectors: User feature vectors (batch_size, user_dim)
            career_vectors: Career feature vectors (batch_size, career_dim)
        
        Returns:
            Predicted scores (batch_size, 1)
        """
        features = self.compute_interaction_features(user_vectors, career_vectors)
        scores = self.network(features)
        return scores


def train_epoch(
    model: CareerReRanker,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Re-ranker model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Training device
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for user_vecs, career_vecs, labels in dataloader:
        user_vecs = user_vecs.to(device)
        career_vecs = career_vecs.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(user_vecs, career_vecs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: CareerReRanker,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Re-ranker model
        dataloader: Validation data loader
        criterion: Loss function
        device: Evaluation device
    
    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for user_vecs, career_vecs, labels in dataloader:
            user_vecs = user_vecs.to(device)
            career_vecs = career_vecs.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(user_vecs, career_vecs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Accuracy (threshold at 0.5)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_reranker(
    train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 5,
    checkpoint_dir: str | Path = "ml/artifacts/models",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> CareerReRanker:
    """
    Train the re-ranker model.
    
    Args:
        train_data: Tuple of (user_vecs, career_vecs, labels)
        val_data: Validation data tuple
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory to save checkpoints
        device: Training device
    
    Returns:
        Trained model
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device)
    LOGGER.info(f"Training on device: {device}")
    
    # Prepare data loaders
    train_user, train_career, train_labels = train_data
    val_user, val_career, val_labels = val_data
    
    train_dataset = TensorDataset(
        torch.from_numpy(train_user),
        torch.from_numpy(train_career),
        torch.from_numpy(train_labels)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_user),
        torch.from_numpy(val_career),
        torch.from_numpy(val_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    user_dim = train_user.shape[1]
    career_dim = train_career.shape[1]
    
    model = CareerReRanker(user_dim=user_dim, career_dim=career_dim)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        LOGGER.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                },
                checkpoint_path / "best_reranker.pt"
            )
            LOGGER.info(f"Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                LOGGER.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    checkpoint = torch.load(checkpoint_path / "best_reranker.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def load_reranker(
    checkpoint_path: str | Path,
    user_dim: int = 914,
    career_dim: int = 515,
    device: str = "cpu"
) -> CareerReRanker:
    """
    Load trained re-ranker from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        user_dim: User vector dimension
        career_dim: Career vector dimension
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = CareerReRanker(user_dim=user_dim, career_dim=career_dim)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    LOGGER.info(f"Loaded model from {checkpoint_path}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load training data
    data_dir = Path("ml/artifacts/training_data")
    
    train_npz = np.load(data_dir / "train.npz")
    val_npz = np.load(data_dir / "val.npz")
    
    train_data = (
        train_npz["user_vectors"],
        train_npz["career_vectors"],
        train_npz["labels"]
    )
    val_data = (
        val_npz["user_vectors"],
        val_npz["career_vectors"],
        val_npz["labels"]
    )
    
    # Train model
    model = train_reranker(train_data, val_data, epochs=50)
    
    LOGGER.info("Training complete!")
