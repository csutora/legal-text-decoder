# Data preprocessing script
# This script handles data loading, cleaning, and transformation from Label Studio JSON exports.

import json
import os
import re
from pathlib import Path
from typing import Optional
import random

from utils import setup_logger
import config

logger = setup_logger(__name__)

# Label mapping from Hungarian to integer scores
LABEL_MAPPING = {
    "1-Nagyon nehezen érthető": 1,
    "2-Nehezen érthető": 2,
    "3-Többé/kevésbé megértem": 3,
    "4-Érthető": 4,
    "5-Könnyen érthető": 5,
}


def extract_label_from_choice(choice: str) -> Optional[int]:
    """
    Extract numeric label from a Label Studio choice string.
    Handles variations in formatting.
    """
    # Direct mapping lookup
    if choice in LABEL_MAPPING:
        return LABEL_MAPPING[choice]

    # Try to extract number from the beginning of the string
    match = re.match(r'^(\d)', choice)
    if match:
        label = int(match.group(1))
        if 1 <= label <= 5:
            return label

    logger.warning(f"Could not parse label from choice: {choice}")
    return None


def parse_label_studio_json(filepath: Path) -> list[dict]:
    """
    Parse a Label Studio JSON export file and extract text-label pairs.

    Returns a list of dicts with keys: 'text', 'label', 'source_file', 'source_folder'
    """
    samples = []
    source_folder = filepath.parent.name
    source_file = filepath.name

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {filepath}: {e}")
        return samples
    except Exception as e:
        logger.error(f"Failed to read file {filepath}: {e}")
        return samples

    # Handle both list and single object formats
    if isinstance(data, dict):
        data = [data]

    for item in data:
        # Skip if no annotations
        annotations = item.get('annotations', [])
        if not annotations:
            continue

        # Get the first annotation (usually there's only one)
        annotation = annotations[0]

        # Skip cancelled annotations
        if annotation.get('was_cancelled', False):
            continue

        # Extract the result
        results = annotation.get('result', [])
        if not results:
            continue

        # Find the choice result
        label = None
        for result in results:
            if result.get('type') == 'choices':
                choices = result.get('value', {}).get('choices', [])
                if choices:
                    label = extract_label_from_choice(choices[0])
                    break

        if label is None:
            continue

        # Extract text - try different possible locations
        text = None
        data_field = item.get('data', {})

        # Common text field names in Label Studio exports
        for text_key in ['text', 'paragraph', 'content', 'sentence']:
            if text_key in data_field:
                text = data_field[text_key]
                break

        if text is None:
            continue

        # Clean text
        text = text.strip()
        if not text:
            continue

        samples.append({
            'text': text,
            'label': label,
            'source_folder': source_folder,
            'source_file': source_file,
        })

    return samples


def load_all_data(data_dir: str) -> list[dict]:
    """
    Load all JSON files from the data directory and its subdirectories.
    """
    data_path = Path(data_dir)
    all_samples = []

    # Find all JSON files
    json_files = list(data_path.glob('**/*.json'))
    logger.info(f"Found {len(json_files)} JSON files in {data_dir}")

    for json_file in json_files:
        # Skip meta.json files (typically Label Studio config)
        if json_file.name == 'meta.json':
            continue

        samples = parse_label_studio_json(json_file)
        if samples:
            logger.info(f"Loaded {len(samples)} samples from {json_file.relative_to(data_path)}")
            all_samples.extend(samples)

    return all_samples


def deduplicate_samples(samples: list[dict]) -> list[dict]:
    """
    Remove duplicate texts, keeping the first occurrence.
    """
    seen_texts = set()
    unique_samples = []

    for sample in samples:
        text_normalized = sample['text'].lower().strip()
        if text_normalized not in seen_texts:
            seen_texts.add(text_normalized)
            unique_samples.append(sample)

    removed = len(samples) - len(unique_samples)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate samples")

    return unique_samples


def split_data(
    samples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split data into train/val/test sets with stratification by label.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    random.seed(seed)

    # Group samples by label for stratified split
    by_label = {}
    for sample in samples:
        label = sample['label']
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(sample)

    train_samples = []
    val_samples = []
    test_samples = []

    for label, label_samples in by_label.items():
        random.shuffle(label_samples)
        n = len(label_samples)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_samples.extend(label_samples[:n_train])
        val_samples.extend(label_samples[n_train:n_train + n_val])
        test_samples.extend(label_samples[n_train + n_val:])

    # Shuffle each split
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def save_dataset(samples: list[dict], filepath: Path):
    """Save samples to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(samples)} samples to {filepath}")


def print_statistics(samples: list[dict], name: str = "Dataset"):
    """Print dataset statistics."""
    logger.info(f"\n{'='*50}")
    logger.info(f"{name} Statistics:")
    logger.info(f"{'='*50}")
    logger.info(f"Total samples: {len(samples)}")

    # Label distribution
    label_counts = {}
    for sample in samples:
        label = sample['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    logger.info("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = 100.0 * count / len(samples)
        logger.info(f"  Label {label}: {count:5d} ({pct:5.1f}%)")

    # Text length statistics
    lengths = [len(s['text']) for s in samples]
    logger.info(f"\nText length (chars):")
    logger.info(f"  Min: {min(lengths)}")
    logger.info(f"  Max: {max(lengths)}")
    logger.info(f"  Mean: {sum(lengths)/len(lengths):.1f}")

    # Source folder distribution
    folder_counts = {}
    for sample in samples:
        folder = sample['source_folder']
        folder_counts[folder] = folder_counts.get(folder, 0) + 1

    logger.info(f"\nData from {len(folder_counts)} source folders")
    logger.info(f"{'='*50}\n")


def preprocess():
    """Main preprocessing function."""
    logger.info("="*60)
    logger.info("Starting data preprocessing...")
    logger.info("="*60)

    # Load all data
    logger.info(f"\nLoading data from {config.DATA_DIR}")
    all_samples = load_all_data(config.DATA_DIR)
    logger.info(f"Total samples loaded: {len(all_samples)}")

    if not all_samples:
        logger.error("No samples found! Check data directory.")
        return

    # Deduplicate
    logger.info("\nRemoving duplicates...")
    all_samples = deduplicate_samples(all_samples)

    # Print overall statistics
    print_statistics(all_samples, "Full Dataset")

    # Split data
    logger.info("Splitting data into train/val/test sets...")
    train_samples, val_samples, test_samples = split_data(
        all_samples,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.RANDOM_SEED
    )

    # Print split statistics
    print_statistics(train_samples, "Training Set")
    print_statistics(val_samples, "Validation Set")
    print_statistics(test_samples, "Test Set")

    # Save processed datasets
    processed_dir = Path(config.DATA_DIR) / 'processed'
    save_dataset(all_samples, processed_dir / 'all_data.json')
    save_dataset(train_samples, processed_dir / 'train.json')
    save_dataset(val_samples, processed_dir / 'val.json')
    save_dataset(test_samples, processed_dir / 'test.json')

    logger.info("="*60)
    logger.info("Data preprocessing complete!")
    logger.info("="*60)


if __name__ == "__main__":
    preprocess()
