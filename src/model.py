# Model architecture for Legal Text Decoder
# Uses Hungarian BERT (HuBERT) for text classification with ordinal cross-entropy loss.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

import config


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Ordinal Cross-Entropy Loss for ordinal classification.

    This loss treats labels as ordinal (ordered), penalizing distant
    misclassifications more than adjacent ones.

    For a K-class ordinal problem, we model K-1 binary classifiers:
    P(Y > k) for k in {1, 2, ..., K-1}

    The loss encourages consistent ordinal predictions where if Y > k,
    then Y > j for all j < k.
    """

    def __init__(self, num_classes: int = config.NUM_LABELS):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal cross-entropy loss.

        Args:
            logits: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,), values in [0, num_classes-1]

        Returns:
            Scalar loss value
        """
        batch_size = logits.size(0)
        device = logits.device

        # Convert to probabilities
        probs = F.softmax(logits, dim=1)

        # Compute cumulative probabilities: P(Y <= k) for k in {0, 1, ..., K-1}
        cumulative_probs = torch.cumsum(probs, dim=1)

        # Create ordinal target encoding
        # For target k, the ordinal encoding is [1, 1, ..., 1, 0, 0, ..., 0]
        # where the first k+1 positions are 1 (Y <= j is true for j >= k)
        ordinal_targets = torch.zeros(batch_size, self.num_classes, device=device)
        for i in range(self.num_classes):
            ordinal_targets[:, i] = (targets <= i).float()

        # Binary cross-entropy for each threshold
        # We use P(Y <= k) as predictions and ordinal_targets as labels
        eps = 1e-7
        cumulative_probs = torch.clamp(cumulative_probs, eps, 1 - eps)

        bce_loss = -(
            ordinal_targets * torch.log(cumulative_probs) +
            (1 - ordinal_targets) * torch.log(1 - cumulative_probs)
        )

        # Average over all thresholds and batch
        loss = bce_loss.mean()

        return loss


class WeightedOrdinalLoss(nn.Module):
    """
    Weighted combination of cross-entropy and ordinal loss.

    This combines standard classification loss with ordinal penalties
    to balance between exact class prediction and ordinal consistency.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_LABELS,
        ordinal_weight: float = 0.5,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ordinal_weight = ordinal_weight
        self.ce_weight = 1.0 - ordinal_weight

        # Standard cross-entropy with optional class weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.ordinal_loss = OrdinalCrossEntropyLoss(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted combination of CE and ordinal loss."""
        ce = self.ce_loss(logits, targets)
        ordinal = self.ordinal_loss(logits, targets)

        return self.ce_weight * ce + self.ordinal_weight * ordinal


class LegalTextClassifier(nn.Module):
    """
    BERT-based classifier for legal text understandability.

    Uses a pretrained Hungarian BERT model (HuBERT) with a classification head.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_labels: int = config.NUM_LABELS,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        super().__init__()
        self.num_labels = num_labels

        # Load pretrained BERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Optionally freeze encoder layers
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, num_labels)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classifier
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Predict class labels (0-indexed)."""
        logits = self.forward(input_ids, attention_mask)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Predict class probabilities."""
        logits = self.forward(input_ids, attention_mask)
        return F.softmax(logits, dim=1)


def compute_class_weights(labels: list[int], num_classes: int = config.NUM_LABELS) -> torch.Tensor:
    """
    Compute balanced class weights from label distribution.

    Args:
        labels: List of integer labels (0-indexed)
        num_classes: Number of classes

    Returns:
        Tensor of class weights
    """
    from collections import Counter
    import numpy as np

    counts = Counter(labels)
    total = len(labels)

    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Test the model
    print("Testing LegalTextClassifier...")

    # Create model
    model = LegalTextClassifier()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    logits = model(input_ids, attention_mask)
    print(f"Output logits shape: {logits.shape}")

    # Test losses
    targets = torch.randint(0, 5, (batch_size,))

    ce_loss = nn.CrossEntropyLoss()
    ordinal_loss = OrdinalCrossEntropyLoss()
    weighted_loss = WeightedOrdinalLoss()

    print(f"\nCross-Entropy Loss: {ce_loss(logits, targets).item():.4f}")
    print(f"Ordinal Loss: {ordinal_loss(logits, targets).item():.4f}")
    print(f"Weighted Ordinal Loss: {weighted_loss(logits, targets).item():.4f}")
