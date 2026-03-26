# troll/__init__.py
"""
Troll 🧌 — ARC-AGI-3 agent harness
Made with ❤️ by Aegis Wizard 🧙‍♂️
MIT License — https://github.com/aegiswizard/troll
"""

from troll.interfaces.agent_wrapper import TrollAgent
from troll.core.config import RunConfig, TrollSettings

__version__ = "1.0.0"
__all__ = ["TrollAgent", "RunConfig", "TrollSettings"]
