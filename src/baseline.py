# Baseline model for Legal Text Decoder
# Uses TF-IDF features with Logistic Regression for text classification.
# This establishes a performance floor before moving to deep learning models.

import json
from pathlib import Path
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from utils import setup_logger
import config

logger = setup_logger(__name__)


def load_data(split: str) -> tuple[list[str], list[int]]:
    """Load preprocessed data split."""
    filepath = Path(config.PROCESSED_DATA_DIR) / f"{split}.json"
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    return texts, labels


def cohen_kappa(y_true, y_pred):
    """Calculate Cohen's Kappa for inter-rater agreement."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def evaluate_model(y_true, y_pred, name: str = "Model"):
    """Evaluate model and print metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    mae = mean_absolute_error(y_true, y_pred)
    kappa = cohen_kappa(y_true, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info(f"{name} Evaluation Results")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy:     {accuracy:.4f}")
    logger.info(f"Macro F1:     {macro_f1:.4f}")
    logger.info(f"MAE:          {mae:.4f}")
    logger.info(f"Cohen's Kappa (quadratic): {kappa:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(f"\n{classification_report(y_true, y_pred)}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"\n{confusion_matrix(y_true, y_pred)}")

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'mae': mae,
        'kappa': kappa
    }


def train_baseline():
    """Train and evaluate baseline models."""
    logger.info("="*60)
    logger.info("Training Baseline Models")
    logger.info("="*60)

    # Load data
    logger.info("\nLoading data...")
    train_texts, train_labels = load_data('train')
    val_texts, val_labels = load_data('val')
    test_texts, test_labels = load_data('test')

    logger.info(f"Train size: {len(train_texts)}")
    logger.info(f"Val size:   {len(val_texts)}")
    logger.info(f"Test size:  {len(test_texts)}")

    # TF-IDF Vectorizer
    logger.info("\nCreating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)

    logger.info(f"TF-IDF features: {X_train.shape[1]}")

    # Baseline 1: Logistic Regression
    logger.info("\n" + "-"*50)
    logger.info("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=config.RANDOM_SEED,
        multi_class='multinomial'
    )
    lr_model.fit(X_train, train_labels)

    lr_val_pred = lr_model.predict(X_val)
    lr_test_pred = lr_model.predict(X_test)

    logger.info("\n--- Logistic Regression (Validation) ---")
    lr_val_metrics = evaluate_model(val_labels, lr_val_pred, "LR Validation")

    logger.info("\n--- Logistic Regression (Test) ---")
    lr_test_metrics = evaluate_model(test_labels, lr_test_pred, "LR Test")

    # Baseline 2: Linear SVM
    logger.info("\n" + "-"*50)
    logger.info("Training Linear SVM...")
    svm_model = LinearSVC(
        max_iter=5000,
        class_weight='balanced',
        random_state=config.RANDOM_SEED,
        dual=True
    )
    svm_model.fit(X_train, train_labels)

    svm_val_pred = svm_model.predict(X_val)
    svm_test_pred = svm_model.predict(X_test)

    logger.info("\n--- Linear SVM (Validation) ---")
    svm_val_metrics = evaluate_model(val_labels, svm_val_pred, "SVM Validation")

    logger.info("\n--- Linear SVM (Test) ---")
    svm_test_metrics = evaluate_model(test_labels, svm_test_pred, "SVM Test")

    # Summary comparison
    logger.info("\n" + "="*60)
    logger.info("BASELINE SUMMARY")
    logger.info("="*60)
    logger.info("\nValidation Results:")
    logger.info(f"  Logistic Regression: Acc={lr_val_metrics['accuracy']:.4f}, F1={lr_val_metrics['macro_f1']:.4f}, MAE={lr_val_metrics['mae']:.4f}")
    logger.info(f"  Linear SVM:          Acc={svm_val_metrics['accuracy']:.4f}, F1={svm_val_metrics['macro_f1']:.4f}, MAE={svm_val_metrics['mae']:.4f}")

    logger.info("\nTest Results:")
    logger.info(f"  Logistic Regression: Acc={lr_test_metrics['accuracy']:.4f}, F1={lr_test_metrics['macro_f1']:.4f}, MAE={lr_test_metrics['mae']:.4f}")
    logger.info(f"  Linear SVM:          Acc={svm_test_metrics['accuracy']:.4f}, F1={svm_test_metrics['macro_f1']:.4f}, MAE={svm_test_metrics['mae']:.4f}")

    # Save best baseline model
    best_model_name = "Logistic Regression" if lr_val_metrics['macro_f1'] >= svm_val_metrics['macro_f1'] else "Linear SVM"
    best_model = lr_model if lr_val_metrics['macro_f1'] >= svm_val_metrics['macro_f1'] else svm_model

    logger.info(f"\nBest baseline model: {best_model_name}")

    # Save model and vectorizer
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(tfidf, models_dir / "baseline_tfidf.joblib")
    joblib.dump(best_model, models_dir / "baseline_model.joblib")
    logger.info(f"Saved baseline model to {models_dir}")

    # Save results
    results = {
        'logistic_regression': {
            'validation': lr_val_metrics,
            'test': lr_test_metrics
        },
        'linear_svm': {
            'validation': svm_val_metrics,
            'test': svm_test_metrics
        },
        'best_model': best_model_name
    }

    with open(models_dir / "baseline_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("="*60)
    logger.info("Baseline training complete!")
    logger.info("="*60)

    return results


if __name__ == "__main__":
    train_baseline()
