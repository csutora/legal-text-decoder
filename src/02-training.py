# Training script for Legal Text Decoder
# Trains a HuBERT-based classifier on Hungarian legal text data.

import json
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from dataset import LegalTextDataset, create_dataloaders
from model import LegalTextClassifier, WeightedOrdinalLoss, compute_class_weights
from utils import setup_logger
import config

logger = setup_logger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def compute_metrics(predictions: list, targets: list) -> dict:
    """Compute evaluation metrics."""
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, cohen_kappa_score

    predictions = np.array(predictions)
    targets = np.array(targets)

    return {
        'accuracy': accuracy_score(targets, predictions),
        'macro_f1': f1_score(targets, predictions, average='macro'),
        'mae': mean_absolute_error(targets, predictions),
        'kappa': cohen_kappa_score(targets, predictions, weights='quadratic')
    }


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device: torch.device
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_targets.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = avg_loss

    return metrics


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device
) -> dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = avg_loss

    return metrics


def train():
    """Main training function."""
    logger.info("="*60)
    logger.info("Starting Training")
    logger.info("="*60)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create output directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(exist_ok=True)

    # Load data
    logger.info("\nLoading data...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_dataset = LegalTextDataset('train', tokenizer=tokenizer)
    val_dataset = LegalTextDataset('val', tokenizer=tokenizer)
    test_dataset = LegalTextDataset('test', tokenizer=tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size: {len(val_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")

    # Compute class weights
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_weights = compute_class_weights(train_labels).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Create model
    logger.info("\nInitializing model...")
    model = LegalTextClassifier(
        model_name=config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        dropout=0.1
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss function
    if config.LOSS_TYPE == "ordinal_cross_entropy":
        criterion = WeightedOrdinalLoss(
            num_classes=config.NUM_LABELS,
            ordinal_weight=0.5,
            class_weights=class_weights
        )
        logger.info("Using Weighted Ordinal Cross-Entropy Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using Cross-Entropy Loss")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='max')

    # Training loop
    logger.info("\nStarting training loop...")
    best_val_f1 = 0
    training_history = []

    for epoch in range(config.EPOCHS):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config.EPOCHS}")
        logger.info(f"{'='*50}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['macro_f1']:.4f}, MAE: {train_metrics['mae']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}, MAE: {val_metrics['mae']:.4f}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        })

        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': {
                    'model_name': config.MODEL_NAME,
                    'num_labels': config.NUM_LABELS,
                    'max_length': config.MAX_LENGTH
                }
            }, models_dir / "best_model.pth")
            logger.info(f"  -> New best model saved! (F1: {best_val_f1:.4f})")

        # Early stopping
        if early_stopping(val_metrics['macro_f1']):
            logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Load best model for final evaluation
    logger.info("\n" + "="*60)
    logger.info("Final Evaluation on Test Set")
    logger.info("="*60)

    checkpoint = torch.load(models_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['macro_f1']:.4f}, MAE: {test_metrics['mae']:.4f}, Kappa: {test_metrics['kappa']:.4f}")

    # Save training history
    history_path = models_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'training_history': training_history,
            'best_val_f1': best_val_f1,
            'test_metrics': test_metrics,
            'config': {
                'model_name': config.MODEL_NAME,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'epochs': config.EPOCHS,
                'loss_type': config.LOSS_TYPE
            }
        }, f, indent=2)

    logger.info(f"\nTraining history saved to {history_path}")
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info("="*60)

    return test_metrics


if __name__ == "__main__":
    train()
