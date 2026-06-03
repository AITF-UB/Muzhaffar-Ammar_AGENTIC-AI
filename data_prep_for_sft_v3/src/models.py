"""
Model configuration — edit this file to switch OpenRouter models.

Browse available models at: https://openrouter.ai/models
"""

# ── Active Model ──────────────────────────────────────────────────────
# This is the model used for ALL generation tasks.
# Change this to any model available on OpenRouter.
MODEL: str = "google/gemini-2.5-flash-lite" 
# MODEL: str = "deepseek/deepseek-v4-flash"


# ── Request Parameters ────────────────────────────────────────────────
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 16000

# ── Presets (uncomment one to switch quickly) ─────────────────────────
# MODEL = "google/gemini-2.5-flash"
# MODEL = "google/gemini-2.5-pro"
# MODEL = "anthropic/claude-sonnet-4"
# MODEL = "openai/gpt-4.1"
# MODEL = "deepseek/deepseek-chat-v3-0324"
# MODEL = "meta-llama/llama-4-maverick"
