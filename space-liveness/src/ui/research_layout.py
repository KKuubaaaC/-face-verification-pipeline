"""Research Gradio UI: `gr.Blocks` layout."""

from __future__ import annotations

import gradio as gr

from src.ui import research_strings
from src.ui.research_handlers import analyze_two_images
from src.vision.embedder import VERIFICATION_THRESHOLD


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=research_strings.BLOCKS_TITLE) as demo:
        with gr.Tabs():
            with gr.Tab(research_strings.TAB_RESEARCH_TITLE):
                gr.Markdown(
                    research_strings.MD_COMPARE_TWO.format(threshold=VERIFICATION_THRESHOLD)
                )

                with gr.Row():
                    img_input_a = gr.Image(
                        label=research_strings.LABEL_IMAGE_A,
                        sources=["upload", "webcam"],
                        type="numpy",
                        height=280,
                    )
                    img_input_b = gr.Image(
                        label=research_strings.LABEL_IMAGE_B,
                        sources=["upload", "webcam"],
                        type="numpy",
                        height=280,
                    )

                model_radio = gr.Radio(
                    choices=[
                        "ArcFace (Baseline)",
                        "Vision Transformer (ViT)",
                        "SwinFace (Swin-T)",
                    ],
                    value="ArcFace (Baseline)",
                    label=research_strings.LABEL_MODEL_CHOICE,
                )

                compare_btn = gr.Button(research_strings.BTN_COMPARE, variant="primary")

                with gr.Row():
                    research_text = gr.Markdown(label=research_strings.LABEL_RESULTS_MD)
                    comparison_img = gr.Image(
                        label=research_strings.LABEL_COMPARISON_IMG,
                        type="numpy",
                        height=200,
                        interactive=False,
                    )

                xai_img = gr.Image(
                    label=research_strings.LABEL_XAI_IMG,
                    type="numpy",
                    height=260,
                    interactive=False,
                )

                compare_btn.click(
                    fn=analyze_two_images,
                    inputs=[img_input_a, img_input_b, model_radio],
                    outputs=[research_text, comparison_img, xai_img],
                )

    return demo
