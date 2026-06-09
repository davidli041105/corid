"""
Centralised configuration for CoRID.

Constants and hyperparameters live here so individual modules don't have
to embed magic strings or numbers. Loaded once at import time.
"""

import os
from dotenv import load_dotenv

# Load .env if present. Silent no-op if missing — caller will hit a clear
# error later when the API key is accessed.
load_dotenv()


# --- iFLYTEK Astron API (OpenAI-compatible) ---
#
# CoRID's LLM is reached through iFLYTEK's 讯飞星辰 MaaS · Astron Token Plan,
# which exposes an OpenAI-compatible chat completions endpoint. We point the
# standard `openai` SDK at the Astron base URL and treat it as a drop-in
# replacement for OpenAI's own API.
#
# `astron-code-latest` is the unified model identifier mandated by the Token
# Plan — the actual underlying model is selected by iFLYTEK behind this
# alias. (See the docs' error-code section, where the recommended fix for
# 401/10404 errors is to confirm model ID is exactly "astron-code-latest".)

ASTRON_API_KEY = os.getenv("ASTRON_API_KEY")
ASTRON_BASE_URL = "https://maas-coding-api.cn-huabei-1.xf-yun.com/v2"

# Model identifier — fixed by the provider; do not change unless the
# Token Plan documentation changes.
MODEL_ASTRON = "astron-code-latest"

# Same model used for both M3 agents (Information Extractor + Rule Inducer)
# and for GEPA's reflect-LM. The implementation split is at the prompt level,
# not the model level.
MODEL_M3_AGENT_1 = MODEL_ASTRON  # Information Extractor (tools + workspace)
MODEL_M3_AGENT_2 = MODEL_ASTRON  # Rule Inducer (no tools, reads agent 1 output)
MODEL_REFLECT_LM = MODEL_ASTRON  # GEPA reflect-LM, used during calibration

# Temperature for M3 reasoning. Low but not zero — some determinism, some
# flexibility for the model to explore phrasing in observations and rules.
M3_TEMPERATURE = 0.2


# --- Context window guidance ---
#
# The Astron Token Plan docs note that `astron-code-latest` routes to one of
# several underlying models (DeepSeek-V3.2, GLM-5, Kimi, etc.) with different
# context limits. Configuring conservatively for the smallest reasonable
# limit avoids 10907/10910 errors (token-count-exceeded). We can raise if we
# observe headroom, but the safe default for now is 96K total context.
MAX_CONTEXT_TOKENS = 96_000

# Per-call output cap. Most of M3's outputs are short structured objects
# (observations, draft rules). 8K is generous but bounded.
MAX_OUTPUT_TOKENS = 8_192
