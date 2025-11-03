import random
import numpy as np
import tensorflow as tf
import logging

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)

logger = logging.getLogger(__name__)

CONFIG = {
    "window_size": 64,
    "test_size": 64, #do not change
    "features": ["OpenChange", "RSI", "Volatility", "MA_20", "ODirection"],

    "lstm_units": [128, 64],
    "dense_units": [32, 16],
    "dropout_rate": 0.4,

    "epochs": 128,
    "batch_size": 16,
    "patience": 16,
    "learning_rate": 0.001,
    "tscv_splits": 7,

    "data_paths": ["yDatas/Bist/*.csv", "yDatas/Test/*.csv"],
    "company_index": 0,

    "plot_history": True,
    "plot_predictions": True,
    "plot_metrics": True
}