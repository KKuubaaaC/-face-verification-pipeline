"""User-visible copy for the Hugging Face Gradio demo."""

HF_BLOCKS_TITLE = "Face verification: reference vs attack set"

HF_FIRST_RUN_NOTE = "*First startup may take about 10-15 seconds.*"

HF_MARKDOWN_INTRO = (
    "### Identity verification and spoof resistance\n"
    "**Reference** images match the enrolled gallery. **Attack** images simulate bypass attempts. "
    "Pick a file and click **Run example**. On the right you see the same frame with a status label "
    "and a face bounding box.\n\n"
    "**Run benchmark** processes both sets, builds a table with columns *set*, *file*, *status*, "
    "*expected*, and *match*, plus a gallery of up to **8** annotated images."
)

LABEL_PERSONA_GROUP = "Set"
LABEL_EXAMPLE_FILE = "File"
BTN_RUN_EXAMPLE = "Run example"
LABEL_INPUT_PREVIEW = "Input"
LABEL_OUTPUT_ANNOTATED = "Output (annotated)"
BTN_RUN_BENCHMARK = "Run benchmark"
BENCHMARK_TABLE_HEADERS = ["set", "file", "status", "expected", "match"]
LABEL_BENCHMARK_RESULTS = "Benchmark results"
BENCHMARK_SUMMARY_PLACEHOLDER = "*Click **Run benchmark** to see the summary and gallery.*"
LABEL_BENCHMARK_SUMMARY = "Summary"
GALLERY_SECTION_MD = "#### Result gallery (up to 8 images)"
LABEL_RESULT_GALLERY = "Annotated images from the first rows of each set"

BENCHMARK_NO_FILES = "**Benchmark:** no files to process (after filtering e.g. screenshots)."
