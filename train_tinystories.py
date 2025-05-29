import torch
from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
from datasets import load_dataset
import wandb
import os
import numpy
import torch.serialization
from bitsandbytes.optim import AdamW8bit

class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=8192):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        # Tokenize single example on the fly (no padding)
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=None,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


def main():
    # Initialize wandb
    wandb.init(project="tinystories-llm-training")

    # 1. Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    print(dataset)

    # 2. Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Prepare on-the-fly tokenized torch datasets
    train_dataset = TinyStoriesDataset(dataset['train'], tokenizer)
    eval_split = 'validation' if 'validation' in dataset else 'test'
    eval_dataset = TinyStoriesDataset(dataset[eval_split], tokenizer)

    # 4. Base collator to pad inputs dynamically
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def collate_fn(features):
        # Pad inputs
        batch = base_collator(features)
        # Use inputs as labels for causal LM
        batch['labels'] = batch['input_ids'].clone()
        return batch

    # 5. Model configuration
    config = MistralConfig(
        hidden_size=384,
        intermediate_size=384*4,
        num_hidden_layers=12,
        num_attention_heads=6,
        num_key_value_heads=6,
        max_position_embeddings=4096,
        sliding_window=384,
        hidden_act="gelu_new",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
    )
        
    # 6. Initialize model
    model = MistralForCausalLM(config)
    print(f"Number of parameters: {model.num_parameters()}")
    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=10000, #valuta ogni x batches
        learning_rate=2e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=1,
        save_steps=10000,
        weight_decay=0.01,
        save_strategy="steps",
        warmup_steps=10000,
        logging_dir="./logs",
        logging_steps=25,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 9. Start training
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # 10. Save best model and tokenizer
    output_dir = "./final_tinystories_model"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Best model saved to {output_dir}.")

    wandb.finish()

if __name__ == '__main__':
    main()
