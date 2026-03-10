"""
config.py — Central Configuration
"""

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")


class Config:
    def __init__(self):
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

        # Groq (cloud LLM)
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        self.GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def validate(self):
        if self.LLM_PROVIDER == "groq" and not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return True


config = Config()
