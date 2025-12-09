# Configuration settings for the Legal Text Decoder project

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = "data"
MODEL_SAVE_PATH = "models/best_model.pth"
PROCESSED_DATA_DIR = "data/processed"

# =============================================================================
# Data Split Ratios
# =============================================================================
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = "SZTAKI-HLT/hubert-base-cc"  # Hungarian BERT
NUM_LABELS = 5  # Understandability scale 1-5
MAX_LENGTH = 512  # Maximum token length (HuBERT max position embeddings)

# =============================================================================
# Training Hyperparameters
# =============================================================================
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1

# =============================================================================
# Regularization
# =============================================================================
DROPOUT = 0.3  # Dropout rate for classification head
FREEZE_ENCODER_LAYERS = 4  # Freeze first N transformer layers (0 = none, HuBERT has 12)

# =============================================================================
# Loss Function
# =============================================================================
# Options: "cross_entropy", "ordinal_cross_entropy", "mse"
LOSS_TYPE = "ordinal_cross_entropy"

# =============================================================================
# Evaluation
# =============================================================================
EVAL_METRICS = ["accuracy", "macro_f1", "mae", "kappa"]

# =============================================================================
# API / Frontend
# =============================================================================
API_HOST = "0.0.0.0"
API_PORT = 8000
GRADIO_PORT = 7860

# =============================================================================
# Logging
# =============================================================================
LOG_DIR = "log"
LOG_FILE = "log/run.log"
