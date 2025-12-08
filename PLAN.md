# Legal Text Decoder - Outstanding Level Implementation Plan

## Project Overview
A system that takes a paragraph of Hungarian legal text and labels its understandability on a scale of 1-5:
- 1: Nagyon nehezen érthető (Very hard to understand)
- 2: Nehezen érthető (Hard to understand)
- 3: Többé/kevésbé megértem (More or less understand)
- 4: Érthető (Understandable)
- 5: Könnyen érthető (Easy to understand)

## Data Summary
- 33 JSON files from 26 student folders in `data/`
- Label Studio export format
- Text field: `data.text`
- Label field: `annotations[0].result[0].value.choices[0]`

---

## Implementation Steps

### Phase 1: Data Pipeline
- [x] Step 1.1: Data preprocessing script (`01-data-preprocessing.py`)
  - Parse all JSON files from different students
  - Extract text and convert Hungarian labels to integers 1-5
  - Remove cancelled annotations
  - Handle missing/empty annotations
  - Create train/val/test splits (stratified)
  - Save as unified format

- [x] Step 1.2: Data exploration notebook (`notebook/01-data-exploration.ipynb`)
  - Label distribution visualization
  - Text length statistics
  - Per-source analysis

- [x] Step 1.3: Label analysis notebook (`notebook/02-label-analysis.ipynb`)
  - Class imbalance analysis
  - Inter-annotator patterns (if applicable)

### Phase 2: Baseline Model
- [x] Step 2.1: Implement TF-IDF + Logistic Regression baseline
  - Simple sklearn pipeline
  - Establish performance floor
  - Results: Accuracy=0.36, Macro F1=0.30, MAE=0.97

### Phase 3: Deep Learning Model
- [x] Step 3.1: Dataset class (`src/dataset.py`)
  - PyTorch Dataset for text classification
  - Tokenization with HuggingFace

- [x] Step 3.2: Model architecture (`src/model.py`)
  - Hungarian BERT: `SZTAKI-HLT/hubert-base-cc`
  - Classification head for 5 classes
  - Ordinal cross-entropy loss implementation

- [x] Step 3.3: Training script (`src/02-training.py`)
  - Ordinal cross-entropy loss (penalizes distant mistakes less)
  - Early stopping
  - Model checkpointing
  - Logging per requirements

- [x] Step 3.4: Evaluation script (`src/03-evaluation.py`)
  - Metrics: Accuracy, Macro F1, MAE, Cohen's Kappa
  - Confusion matrix
  - Per-class precision/recall

### Phase 4: Inference & API
- [x] Step 4.1: Inference script (`src/04-inference.py`)
  - Load trained model
  - Predict on new text

- [x] Step 4.2: Gradio frontend (`src/app.py`)
  - Simple web interface
  - Text input -> understandability score

### Phase 5: Containerization & Documentation
- [x] Step 5.1: Update Dockerfile
  - GPU support (CUDA)
  - All dependencies

- [x] Step 5.2: Update requirements.txt
  - transformers, gradio, etc.

- [x] Step 5.3: Update README.md
  - Project description
  - Docker instructions
  - Solution description

- [ ] Step 5.4: Generate run.log
  - Full pipeline execution log (requires running training with GPU)

---

## Technical Decisions

### Loss Function
- **Ordinal Cross-Entropy**: Treats mistakes between adjacent classes (e.g., 3→4) as smaller errors than distant mistakes (e.g., 2→5)
- Alternative: Standard cross-entropy with class weights

### Model
- Primary: `SZTAKI-HLT/hubert-base-cc` (Hungarian BERT)
- Fallback: `xlm-roberta-base` (multilingual)

### Evaluation Metrics
- **Primary**: Macro F1-score, MAE (mean absolute error for ordinal)
- **Secondary**: Accuracy, Cohen's Kappa, confusion matrix

### Frontend
- Gradio (simple, effective, easy to containerize)

---

## Hardware
- NVIDIA RTX 5090
- NixOS (use `nix-shell -p <package> --run "<command>"`)

---

## Priority
Pipeline correctness first, then performance optimization.
