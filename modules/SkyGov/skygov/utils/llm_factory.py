"""Shared LLM client factory — loads API key from .env or environment."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_llm_client(api_key: Optional[str] = None):
    """Create an OpenAI-compatible client for DeepSeek.

    Priority: explicit arg > env var > .env file.
    Returns None if no key is found (agents fall back to mock mode).
    """
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        try:
            from dotenv import load_dotenv
            env_candidates = [
                Path(__file__).resolve().parents[2] / ".env",   # modules/SkyGov/.env
                Path(__file__).resolve().parents[4] / ".env",   # repo root .env
            ]
            for p in env_candidates:
                if p.exists():
                    load_dotenv(p)
                    api_key = os.getenv("DEEPSEEK_API_KEY")
                    if api_key:
                        break
        except ImportError:
            pass

    if not api_key:
        logger.warning("No DEEPSEEK_API_KEY found — agents will use mock mode")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        logger.info("DeepSeek LLM client initialized (real API mode)")
        return client
    except Exception as e:
        logger.error("Failed to create LLM client: %s", e)
        return None
