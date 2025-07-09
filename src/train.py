# src/train.py

from datasets import load_from_disk
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def main():
    # Load tokenized data
    tokenized_dataset = load_from_disk("./data/tokenized_wikitext2")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./models",
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    # Train the model
    trainer.train()

    # Save final model and tokenizer
    model.save_pretrained("./models/gpt2-finetuned")
    tokenizer.save_pretrained("./models/gpt2-finetuned")

if __name__ == "__main__":
    main()
