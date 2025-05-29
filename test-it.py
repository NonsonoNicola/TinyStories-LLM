import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Mistral model tester: load a model and autocomplete prompts.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=50,
        help="Maximum number of tokens to generate")
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for generation")
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from model_dir onto {device}...")

    # Attempt to load local tokenizer, fallback to official checkpoint if missing
    try:
        tokenizer = AutoTokenizer.from_pretrained("model_dir", local_files_only=True)
    except Exception:
        print("Local tokenizer not found. Downloading base Mistral tokenizer and caching locally...")
        base_name = "mistralai/Mistral-7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        tokenizer.save_pretrained("model_dir")

    # Load model weights
    model = AutoModelForCausalLM.from_pretrained("model_dir", local_files_only=True)
    model.to(device)
    model.eval()

    print("\nModel loaded. Enter your prompts below (press Enter on empty line to exit).\n")
    while True:
        prompt = input(">> ")
        if not prompt.strip():
            print("Exiting interactive session.")
            break

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(device)

        # Generate continuation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )

        # Decode and display only the generated part
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = full_text[len(prompt):]
        print(f"{continuation}\n")

if __name__ == "__main__":
    main()
