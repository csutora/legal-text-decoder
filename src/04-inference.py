# Inference script for Legal Text Decoder
# This script runs the model on new, unseen text data.

from pathlib import Path
from typing import Union

import torch
from transformers import AutoTokenizer

from model import LegalTextClassifier
from utils import setup_logger
import config

logger = setup_logger(__name__)

# Label descriptions (1-indexed, human-readable)
LABEL_DESCRIPTIONS = {
    1: "Nagyon nehezen érthető (Very hard to understand)",
    2: "Nehezen érthető (Hard to understand)",
    3: "Többé/kevésbé megértem (Somewhat understandable)",
    4: "Érthető (Understandable)",
    5: "Könnyen érthető (Easy to understand)"
}


class LegalTextPredictor:
    """Predictor class for legal text understandability."""

    def __init__(
        self,
        model_path: str = "models/best_model.pth",
        device: str = None
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model, self.model_config = self._load_model(model_path)
        self.model.eval()

        # Load tokenizer
        model_name = self.model_config.get('model_name', config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = self.model_config.get('max_length', config.MAX_LENGTH)

        logger.info("Predictor initialized successfully")

    def _load_model(self, model_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        model_config = checkpoint.get('config', {})
        model_name = model_config.get('model_name', config.MODEL_NAME)
        num_labels = model_config.get('num_labels', config.NUM_LABELS)

        model = LegalTextClassifier(
            model_name=model_name,
            num_labels=num_labels
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        return model, model_config

    def predict(self, text: str) -> dict:
        """
        Predict understandability of a single text.

        Args:
            text: Legal text to analyze

        Returns:
            Dictionary with prediction results:
            - label: Predicted label (1-5)
            - description: Human-readable label description
            - confidence: Confidence score (probability of predicted class)
            - probabilities: All class probabilities
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).squeeze()
            predicted_class = torch.argmax(probs).item()

        # Convert to 1-indexed label
        label = predicted_class + 1
        confidence = probs[predicted_class].item()
        all_probs = {i+1: probs[i].item() for i in range(5)}

        return {
            'label': label,
            'description': LABEL_DESCRIPTIONS[label],
            'confidence': confidence,
            'probabilities': all_probs
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Predict understandability of multiple texts.

        Args:
            texts: List of legal texts to analyze

        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def predict():
    """Interactive inference function."""
    logger.info("="*60)
    logger.info("Legal Text Decoder - Inference")
    logger.info("="*60)

    # Check if model exists
    model_path = Path("models") / "best_model.pth"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run training first.")
        return

    # Initialize predictor
    predictor = LegalTextPredictor(str(model_path))

    # Example texts for demonstration
    example_texts = [
        "A szerződés létrejöttével a vevő vállalja, hogy a vételárat megfizeti.",
        "Az üzletszabályzat módosítása esetén az e tárgyban közzétett hirdetmény tartalmazza a megváltoztatott szabályokat és a hatálybalépés napját is.",
        "A Vtv. 69. § 25. pontja alapján a vasúti társaság Üzletszabályzatának jóváhagyása a Vtv. szerinti vasúti igazgatási szerv hatáskörébe tartozik.",
    ]

    logger.info("\nRunning predictions on example texts:\n")

    for i, text in enumerate(example_texts, 1):
        logger.info(f"Text {i}:")
        logger.info(f"  \"{text[:100]}...\"" if len(text) > 100 else f"  \"{text}\"")

        result = predictor.predict(text)

        logger.info(f"  Prediction: {result['label']} - {result['description']}")
        logger.info(f"  Confidence: {result['confidence']:.2%}")
        logger.info(f"  All probabilities:")
        for label, prob in result['probabilities'].items():
            logger.info(f"    Label {label}: {prob:.2%}")
        logger.info("")

    logger.info("="*60)
    logger.info("Inference complete!")
    logger.info("="*60)


if __name__ == "__main__":
    predict()
