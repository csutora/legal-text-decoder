# Legal Text Decoder

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Márton Csutora
- **Aiming for +1 Mark**: Yes

### Solution Description

This project implements a **Legal Text Decoder** - a system that analyzes Hungarian legal texts (ÁSZF - Általános Szerződési Feltételek / General Terms and Conditions) and predicts their understandability on a scale of 1-5:

| Score | Hungarian | English |
|-------|-----------|---------|
| 1 | Nagyon nehezen érthető | Very hard to understand |
| 2 | Nehezen érthető | Hard to understand |
| 3 | Többé/kevésbé megértem | Somewhat understandable |
| 4 | Érthető | Understandable |
| 5 | Könnyen érthető | Easy to understand |

#### Model Architecture

The solution uses a **fine-tuned Hungarian BERT model (HuBERT)** for text classification:

- **Base Model**: `SZTAKI-HLT/hubert-base-cc` - A Hungarian BERT model pretrained on a large Hungarian corpus
- **Architecture**: BERT encoder + Classification head (768 → 5 classes)
- **Loss Function**: **Ordinal Cross-Entropy** - A custom loss that treats labels as ordinal, penalizing distant misclassifications more than adjacent ones (e.g., predicting 2 when true is 4 is worse than predicting 3 when true is 4)
- **Class Weighting**: Balanced class weights to handle label imbalance

#### Training Methodology

1. **Data Preprocessing**: Aggregated 3,397 labeled samples from 33 JSON files (Label Studio exports) from 25 different annotators
2. **Stratified Split**: 80% train / 10% validation / 10% test with stratification by label
3. **Training**: Fine-tuning with AdamW optimizer, linear warmup scheduler, early stopping (patience=5)
4. **Evaluation**: Accuracy, Macro F1, MAE, Cohen's Kappa (quadratic weighted)

#### Results Summary

| Model | Accuracy | Macro F1 | MAE |
|-------|----------|----------|-----|
| Baseline (TF-IDF + LogReg) | 0.36 | 0.30 | 0.97 |
| HuBERT + Ordinal Loss | TBD | TBD | TBD |

### Extra Credit Justification

This project qualifies for the +1 mark based on:

1. **Complete Pipeline**: Full implementation from data preprocessing to web deployment
2. **Advanced Loss Function**: Custom ordinal cross-entropy loss that respects the ordinal nature of understandability ratings
3. **ML as a Service**: Gradio-based web interface for interactive text analysis
4. **Comprehensive Evaluation**: Multiple metrics including Cohen's Kappa for ordinal agreement
5. **Multi-source Data Handling**: Robust parsing of Label Studio exports from multiple annotators with varying formats

### Data Preparation

The data is collected from Hungarian legal documents (ÁSZF) labeled by multiple students using Label Studio. The data files are located in the `data/` directory, with each student's contributions in separate folders.

**Data Format**: Label Studio JSON export with the following structure:
```json
{
  "data": {"text": "Legal text paragraph..."},
  "annotations": [{
    "result": [{
      "value": {"choices": ["4-Érthető"]},
      "type": "choices"
    }]
  }]
}
```

The preprocessing script (`src/01-data-preprocessing.py`) handles:
- Parsing multiple JSON formats from different annotators
- Extracting text and labels (Hungarian label strings → integers 1-5)
- Removing cancelled annotations and duplicates
- Stratified train/val/test splitting

### Docker Instructions

This project is containerized using Docker with GPU support.

#### Build

```bash
# CPU version
docker build -t legal-text-decoder .

# GPU version (requires NVIDIA Docker runtime)
docker build --target gpu -t legal-text-decoder:gpu .
```

#### Run Full Pipeline

```bash
# Run with data mounted, capture logs (CPU version)
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models legal-text-decoder > log/run.log 2>&1

# GPU version (requires NVIDIA Container Toolkit)
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models legal-text-decoder:gpu > log/run.log 2>&1
```

> **Note**: For GPU support in Docker, you need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed. Alternatively, run training directly on your host machine (see [Local Development](#local-development)).

#### Run Gradio Web Interface

```bash
# Start the web interface on port 7860
docker run -p 7860:7860 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
    legal-text-decoder python src/app.py
```

Then open http://localhost:7860 in your browser.

#### Run Individual Scripts

```bash
# Data preprocessing only
docker run -v $(pwd)/data:/app/data legal-text-decoder python src/01-data-preprocessing.py

# Training only
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
    legal-text-decoder python src/02-training.py

# Evaluation only
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
    legal-text-decoder python src/03-evaluation.py

# Inference on examples
docker run -v $(pwd)/models:/app/models legal-text-decoder python src/04-inference.py
```

### File Structure and Functions

```
legal-text-decoder/
├── src/
│   ├── 01-data-preprocessing.py  # Data loading, cleaning, and splitting
│   ├── 02-training.py            # Model training with ordinal loss
│   ├── 03-evaluation.py          # Comprehensive model evaluation
│   ├── 04-inference.py           # Inference on new texts
│   ├── app.py                    # Gradio web application
│   ├── baseline.py               # TF-IDF + LogReg baseline model
│   ├── config.py                 # Hyperparameters and paths
│   ├── dataset.py                # PyTorch Dataset class
│   ├── model.py                  # HuBERT classifier + ordinal loss
│   └── utils.py                  # Logging utilities
│
├── notebook/
│   ├── 01-data-exploration.ipynb # EDA and visualization
│   └── 02-label-analysis.ipynb   # Label distribution analysis
│
├── data/
│   ├── <student_id>/             # Raw Label Studio exports
│   └── processed/                # Preprocessed train/val/test splits
│       ├── train.json
│       ├── val.json
│       ├── test.json
│       └── all_data.json
│
├── models/
│   ├── best_model.pth            # Best trained model checkpoint
│   ├── baseline_model.joblib     # Baseline model
│   └── training_history.json     # Training metrics history
│
├── log/
│   └── run.log                   # Pipeline execution log
│
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
├── run.sh                        # Full pipeline script
├── PLAN.md                       # Implementation plan
└── README.md                     # This file
```

### Configuration

Key hyperparameters in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| MODEL_NAME | SZTAKI-HLT/hubert-base-cc | Hungarian BERT model |
| NUM_LABELS | 5 | Understandability classes |
| MAX_LENGTH | 512 | Max token sequence length |
| BATCH_SIZE | 16 | Training batch size |
| LEARNING_RATE | 2e-5 | AdamW learning rate |
| EPOCHS | 20 | Maximum training epochs |
| EARLY_STOPPING_PATIENCE | 5 | Early stopping patience |
| LOSS_TYPE | ordinal_cross_entropy | Loss function type |

### Requirements

- Python 3.10+
- PyTorch 2.3+
- Transformers 4.40+
- CUDA 12.1+ (for GPU training)
- See `requirements.txt` for full dependencies

### Local Development

#### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Running the Pipeline

```bash
# Run full pipeline
bash run.sh

# Or run individual scripts:
python src/01-data-preprocessing.py  # Data preprocessing
python src/baseline.py               # Baseline model
python src/02-training.py            # Train HuBERT model
python src/03-evaluation.py          # Evaluate model
python src/04-inference.py           # Run inference examples
python src/app.py                    # Start Gradio web interface
```

#### Running the Web Interface

```bash
python src/app.py
# Open http://localhost:7860 in your browser
```
