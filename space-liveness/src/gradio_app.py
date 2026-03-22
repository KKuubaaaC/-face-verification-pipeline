"""
Face Verification & Anti-Spoofing — Gradio web interface (Hugging Face Space).

Layout lives in ``src/ui/hf_layout.py``; business logic in ``src/ui/hf_handlers.py``.
"""

from __future__ import annotations

import os

from src.ui.hf_handlers import DB_REAL, logger
from src.ui.hf_layout import build_ui

__all__ = ["build_ui", "launch", "logger"]


def launch() -> None:
    logger.info("Starting Face Verification & Anti-Spoofing.")
    logger.info("Reference DB : %s", DB_REAL)
    demo = build_ui()
    if not os.environ.get("SPACE_ID"):
        port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
        demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)


if __name__ == "__main__":
    launch()
