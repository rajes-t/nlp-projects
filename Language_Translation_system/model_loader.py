
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Get the project directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Local NLLB model directory
MODEL_DIR = PROJECT_ROOT / "models" / "nllb-200-distilled-600M"


def load_translation_model():

    # Check whether GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on: {device}")

    # Select model precision
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Load tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    # Load model weights from local directory
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        dtype=dtype,
        local_files_only=True
    )

    # Move model to selected device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    print("NLLB model loaded successfully!")

    return tokenizer, model, device


if __name__ == "__main__":

    tokenizer, model, device = load_translation_model()

    print("Device:", device)
    print("Model type:", type(model).__name__)
