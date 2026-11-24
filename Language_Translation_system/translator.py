
from model_loader import load_translation_model
import torch

# Load the tokenizer and model
tokenizer, model, device = load_translation_model()


# Source and target language codes
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "tam_Taml"


def translate_text(text):

    # Set the source language
    tokenizer.src_lang = SOURCE_LANGUAGE

    # Convert input sentence into token IDs
    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    # Move input tensors to GPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Get the target language token ID
    target_language_id = tokenizer.convert_tokens_to_ids(
        TARGET_LANGUAGE
    )

    # Generate the translation
    with torch.inference_mode():

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_language_id,
            num_beams=4,
            max_length=128
        )

    # Convert generated token IDs into readable text
    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


if __name__ == "__main__":

    sentence = "Hi Rajesh , i am your assistant. How can i help you ?"

    translation = translate_text(sentence)

    print("English:", sentence)
    print("Tamil:", translation)
