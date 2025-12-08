# Model evaluation script
# This script evaluates the trained model on the test set and generates detailed metrics.

import json
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)

from dataset import LegalTextDataset
from model import LegalTextClassifier
from utils import setup_logger
import config

logger = setup_logger(__name__)


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint.get('config', {})
    model_name = model_config.get('model_name', config.MODEL_NAME)
    num_labels = model_config.get('num_labels', config.NUM_LABELS)

    model = LegalTextClassifier(
        model_name=model_name,
        num_labels=num_labels
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, model_config


def evaluate_model(model, dataloader, device):
    """Run evaluation and collect predictions."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_predictions), np.array(all_targets), np.array(all_probs)


def compute_all_metrics(predictions, targets):
    """Compute comprehensive evaluation metrics."""
    # Basic metrics
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average='macro')
    weighted_f1 = f1_score(targets, predictions, average='weighted')
    mae = mean_absolute_error(targets, predictions)
    kappa = cohen_kappa_score(targets, predictions, weights='quadratic')

    # Per-class metrics
    report = classification_report(targets, predictions, output_dict=True)
    conf_matrix = confusion_matrix(targets, predictions)

    # Convert labels back to 1-5 scale for display
    label_names = {
        0: '1-Very Hard',
        1: '2-Hard',
        2: '3-Moderate',
        3: '4-Easy',
        4: '5-Very Easy'
    }

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'mae': mae,
        'kappa': kappa,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'label_names': label_names
    }


def evaluate():
    """Main evaluation function."""
    logger.info("="*60)
    logger.info("Model Evaluation")
    logger.info("="*60)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model_path = Path("models") / "best_model.pth"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run training first.")
        return

    logger.info(f"\nLoading model from {model_path}")
    model, model_config = load_model(model_path, device)
    logger.info(f"Model config: {model_config}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.get('model_name', config.MODEL_NAME))

    # Create test dataloader
    logger.info("\nLoading test data...")
    test_dataset = LegalTextDataset('test', tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    logger.info(f"Test set size: {len(test_dataset)}")

    # Run evaluation
    logger.info("\nRunning evaluation...")
    predictions, targets, probs = evaluate_model(model, test_loader, device)

    # Compute metrics
    metrics = compute_all_metrics(predictions, targets)

    # Print results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)

    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")
    logger.info(f"  MAE:               {metrics['mae']:.4f}")
    logger.info(f"  Cohen's Kappa (Q): {metrics['kappa']:.4f}")

    logger.info(f"\nPer-Class Metrics:")
    for label_idx in range(5):
        label_str = str(label_idx)
        if label_str in metrics['classification_report']:
            class_metrics = metrics['classification_report'][label_str]
            label_name = metrics['label_names'][label_idx]
            logger.info(f"  {label_name}:")
            logger.info(f"    Precision: {class_metrics['precision']:.4f}")
            logger.info(f"    Recall:    {class_metrics['recall']:.4f}")
            logger.info(f"    F1-score:  {class_metrics['f1-score']:.4f}")
            logger.info(f"    Support:   {class_metrics['support']}")

    logger.info(f"\nConfusion Matrix:")
    conf_matrix = np.array(metrics['confusion_matrix'])
    logger.info(f"  Predicted →")
    logger.info(f"  True ↓     1     2     3     4     5")
    for i, row in enumerate(conf_matrix):
        logger.info(f"       {i+1}  {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d} {row[4]:5d}")

    # Save results
    results_path = Path("models") / "evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json_metrics = {
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'weighted_f1': float(metrics['weighted_f1']),
            'mae': float(metrics['mae']),
            'kappa': float(metrics['kappa']),
            'confusion_matrix': metrics['confusion_matrix'],
            'per_class': {}
        }
        for label_idx in range(5):
            label_str = str(label_idx)
            if label_str in metrics['classification_report']:
                json_metrics['per_class'][metrics['label_names'][label_idx]] = metrics['classification_report'][label_str]

        json.dump(json_metrics, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("="*60)
    logger.info("Evaluation complete!")
    logger.info("="*60)

    return metrics


if __name__ == "__main__":
    evaluate()
