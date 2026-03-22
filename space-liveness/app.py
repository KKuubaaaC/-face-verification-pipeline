import os

from src.gradio_app import build_ui

demo = build_ui()

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
