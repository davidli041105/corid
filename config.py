"""
Centralised configuration for CoRID.

Constants and hyperparameters live here so individual modules don't have
to embed magic strings or numbers. Loaded once at import time.
"""

import os
from dotenv import load_dotenv

# Load .env if present. Silent no-op if missing — caller will hit a clear
# error later when DEEPSEEK_API_KEY is accessed.
load_dotenv()


# --- DeepSeek API ---

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Model choices. Using V4-Pro everywhere during initial build to remove
# model capability as a variable while debugging. Can switch M3 to
# V4-Flash later for cost reduction once the pipeline is stable.
MODEL_M3 = "deepseek-v4-pro"
MODEL_REFLECT_LM = "deepseek-v4-pro"  # for GEPA later

# Whether to enable the model's "thinking mode" (separate reasoning channel).
# Off for now — simpler trace handling. Can flip on later as an experiment.
USE_THINKING_MODE = False


# --- Misc ---

# Temperature for M3. Low but not zero — some determinism, some flexibility.
M3_TEMPERATURE = 0.2
