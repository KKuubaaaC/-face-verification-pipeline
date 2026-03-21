"""
HF Spaces: musi wywołać demo.launch() także gdy SPACE_ID jest ustawione.
Warunek `not SPACE_ID` powodował pominięcie launch → restart pętli / brak serwera.
"""

from __future__ import annotations

import os

from src.main import build_ui

demo = build_ui()

if __name__ == "__main__":
    # Port z env (HF ustawia GRADIO_SERVER_PORT / PORT); fallback 7860 tylko lokalnie.
    _port_raw = os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("PORT") or "7860"
    _port = int(_port_raw)
    demo.launch(server_name="0.0.0.0", server_port=_port, show_error=True)
