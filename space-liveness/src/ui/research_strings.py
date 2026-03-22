"""User-visible copy for the research Gradio UI."""

TAB_RESEARCH_TITLE = "Research: face embeddings"
BLOCKS_TITLE = "Research: face embeddings"

MD_COMPARE_TWO = (
    "### Compare two face images\n"
    "Upload two face images. We compute the cosine distance between them "
    "and compare it to the EER threshold {threshold} (ArcFace)."
)

LABEL_IMAGE_A = "Image A"
LABEL_IMAGE_B = "Image B"
LABEL_MODEL_CHOICE = "Embedding model"
BTN_COMPARE = "Compare faces"
LABEL_RESULTS_MD = "Results"
LABEL_COMPARISON_IMG = "Aligned crops + distance"
LABEL_XAI_IMG = "Model explainability (attention / XAI)"

MSG_UPLOAD_BOTH = "Upload both images."
MSG_NO_FACE_A = "No face detected on image A."
MSG_NO_FACE_B = "No face detected on image B."
INFO_LOADING_SWINFACE = "Loading SwinFace (first run may take 10-15s)..."
INFO_LOADING_VIT = "Loading ViT model..."

THRESHOLD_NOTE_VIT = "no EER threshold for ViT - indicative only"
THRESHOLD_NOTE_ARCFACE = "EER threshold: `{threshold}`"
THRESHOLD_NOTE_SWINFACE = "ArcFace EER threshold: `{threshold}` (indicative for SwinFace)"

ERR_ANALYZE_PREFIX = "Error: "

BLANK_OVERLAY_TEXT = "No image"
