"""Hugging Face Gradio demo: `gr.Blocks` layout and event wiring."""

from __future__ import annotations

import gradio as gr

from src.ui import hf_strings
from src.ui.hf_handlers import (
    UI_PERSONA_ATTACK,
    UI_PERSONA_REFERENCE,
    _file_choices_for_persona,
    _refresh_file_dropdown,
    run_benchmark_all_examples,
    run_example_verification,
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=hf_strings.HF_BLOCKS_TITLE) as demo:
        gr.Markdown(hf_strings.HF_FIRST_RUN_NOTE)
        gr.Markdown(hf_strings.HF_MARKDOWN_INTRO)

        _files0 = _file_choices_for_persona(UI_PERSONA_REFERENCE)
        with gr.Row():
            persona_radio = gr.Radio(
                choices=[UI_PERSONA_REFERENCE, UI_PERSONA_ATTACK],
                value=UI_PERSONA_REFERENCE,
                label=hf_strings.LABEL_PERSONA_GROUP,
            )
            example_file = gr.Dropdown(
                choices=_files0,
                value=(_files0[0] if _files0 else None),
                label=hf_strings.LABEL_EXAMPLE_FILE,
            )
            run_example_btn = gr.Button(hf_strings.BTN_RUN_EXAMPLE, variant="primary")

        with gr.Row():
            example_input_image = gr.Image(
                label=hf_strings.LABEL_INPUT_PREVIEW,
                type="numpy",
                interactive=False,
                height=360,
            )
            example_output_image = gr.Image(
                label=hf_strings.LABEL_OUTPUT_ANNOTATED,
                type="numpy",
                interactive=False,
                height=360,
            )

        gr.Markdown("---")
        run_benchmark_btn = gr.Button(hf_strings.BTN_RUN_BENCHMARK, variant="secondary")
        benchmark_table = gr.Dataframe(
            headers=hf_strings.BENCHMARK_TABLE_HEADERS,
            datatype=["str", "str", "str", "str", "bool"],
            interactive=False,
            wrap=True,
            label=hf_strings.LABEL_BENCHMARK_RESULTS,
        )
        benchmark_summary = gr.Markdown(
            value=hf_strings.BENCHMARK_SUMMARY_PLACEHOLDER,
            label=hf_strings.LABEL_BENCHMARK_SUMMARY,
        )

        gr.Markdown(hf_strings.GALLERY_SECTION_MD)
        result_gallery = gr.Gallery(
            label=hf_strings.LABEL_RESULT_GALLERY,
            columns=4,
            height=480,
            object_fit="contain",
            show_label=True,
        )

        persona_radio.change(
            fn=_refresh_file_dropdown,
            inputs=[persona_radio],
            outputs=[example_file],
        )
        run_example_btn.click(
            fn=run_example_verification,
            inputs=[persona_radio, example_file],
            outputs=[example_input_image, example_output_image],
        )
        run_benchmark_btn.click(
            fn=run_benchmark_all_examples,
            outputs=[benchmark_table, benchmark_summary, result_gallery],
        )

    return demo
