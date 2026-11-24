

from pathlib import Path
from huggingface_hub import snapshot_download


# Hugging Face model repository
MODEL_ID = "facebook/nllb-200-distilled-600M"


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents

print(PROJECT_ROOT)


# Define the local model directory
MODEL_DIR = PROJECT_ROOT / "models" / "nllb-200-distilled-600M"


# Download the model files into the local directory
snapshot_download(repo_id=MODEL_ID,local_dir=str(MODEL_DIR)
)


print("Model downloaded successfully!")
print(f"Model location: {MODEL_DIR}")
