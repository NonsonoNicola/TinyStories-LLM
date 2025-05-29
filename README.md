# 🧸 TinyStories LLM - A Miniature Language Model for Storytelling

A 40M parameter decoder-only transformer trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), designed to generate short, coherent children's stories. Inspired by the [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1) architecture, built and trained from scratch by a self-taught student exploring large language models.

---

## 🧠 Overview

This project was a hands-on deep dive into how LLMs are built and trained, aiming to answer a simple question: *Can I train a small transformer model to tell simple stories?*

Built using:
- Hugging Face Transformers
- PyTorch
- Bitsandbytes (for 8-bit optimizer)
- Mistral tokenizer + custom architecture
- Trained and tracked with Weights & Biases

---

## 🏗️ Model Architecture

| Component               | Value            |
|------------------------|------------------|
| Layers                 | 12               |
| Hidden Size            | 384              |
| Attention Heads        | 6                |
| Parameters             | ~40 million      |
| Sequence Length        | 4096 tokens      |
| Sliding Window         | 384              |
| Activation             | GELU (new)       |
| Norm                   | RMSNorm          |
| Tokenizer              | Mistral (BPE)    |

---

## 🧪 Training Setup

| Detail                        | Value                        |
|------------------------------|------------------------------|
| Dataset                      | `roneneldan/TinyStories`     |
| Tokenization                 | On-the-fly, dynamic padding  |
| Optimizer                    | AdamW (8-bit) via bitsandbytes |
| Learning Rate                | 2e-5                         |
| Batch Size                   | 6                            |
| Max Length                   | 4096 tokens                  |
| Evaluation Strategy          | Every 10,000 steps           |
| Early Stopping               | After 2 non-improving evals  |
| Logging                      | Weights & Biases             |
| Mixed Precision              | FP16                         |
| Total Training Time          | ~16 hours                    |
| Hardware                     | RTX 2070 8GB GPU             |

---

## 📈 Results

While small in size, the model can consistently generate grammatically correct and logically coherent short story snippets.

### 🔹 Example Output

> **Prompt:** "Once upon a time, a rabbit lived in a quiet forest."
>
> **Model:** "...He liked to hop around and play in the grass. One day, he found a shiny red apple..."

While it's not competitive with large-scale models, this project demonstrates that even small, resource-efficient transformers can learn meaningful language patterns with proper architecture and training.

---

## 💡 What I Learned

- Building a transformer model from a config using Hugging Face
- Tokenizing datasets on-the-fly with attention masks
- Using Trainer API for custom models
- Managing training stability and resource limits (8GB GPU)
- Logging metrics and handling early stopping with WandB

---

## 🚧 Future Work

- Quantize the model for faster inference
- Improve long-term story coherence
- Fine-tune a larger variant on TinyStories+
- Add an interactive Hugging Face Space or Colab demo
- Evaluate using automatic metrics (e.g., BLEU, n-gram diversity)

---

## 🙏 Acknowledgments

- [Ronen Eldan](https://huggingface.co/datasets/roneneldan/TinyStories) for the TinyStories dataset  
- [Mistral](https://huggingface.co/mistralai) for tokenizer + architecture inspiration  
- [Hugging Face](https://huggingface.co/) for the ecosystem  
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) for 8-bit optimizer support  
- [Weights & Biases](https://wandb.ai/) for tracking training

---

## 🧑‍💻 Author

Built by a self-taught 18-year-old computer science student passionate about machine learning and AI systems.
If you'd like to collaborate or just connect, feel free to reach out!
