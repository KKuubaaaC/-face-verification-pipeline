"""
Gradio UI for face verification with liveness (research tab).

Layout: ``src/ui/research_layout.py``. Handlers: ``src/ui/research_handlers.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.ui.research_layout import build_ui  # noqa: E402

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css="",
    )
