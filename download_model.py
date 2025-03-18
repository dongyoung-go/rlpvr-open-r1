from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model name
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"

def download_model():
    print(f"Downloading model: {MODEL_NAME}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype="auto",
        device_map="auto"  # Automatically assign model to available GPUs
    )

    print(f"Model downloaded")

if __name__ == "__main__":
    download_model()

