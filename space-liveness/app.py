import os

from src.gradio_app import build_ui

demo = build_ui()

if __name__ == "__main__":
    _port_raw = os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("PORT") or "7860"
    port = int(_port_raw)
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
