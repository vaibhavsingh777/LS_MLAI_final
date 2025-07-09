# data/tokenize_data.py

from datasets import load_from_disk
from transformers import AutoTokenizer

def tokenize_function(example, tokenizer, max_length=128):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

def main():
    # Load the filtered dataset
    dataset = load_from_disk("./data/filtered_wikitext2")
    print("Loaded filtered dataset.")

    # Load GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS for GPT-2

    # Tokenize the dataset
    def tokenize_batch(batch):
        return tokenize_function(batch, tokenizer)

    print("Tokenizing dataset (this may take a few minutes)...")
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],  # Remove original text to save space
        desc="Tokenizing"
    )

    # Show a tokenized sample
    for split in ['train', 'validation', 'test']:
        print(f"\n--- Tokenized {split} sample ---")
        print({k: tokenized_dataset[split][0][k] for k in ['input_ids', 'attention_mask']})
        break

    # Save tokenized dataset
    tokenized_dataset.save_to_disk("./data/tokenized_wikitext2")
    print("\nTokenized dataset saved to ./data/tokenized_wikitext2")

if __name__ == "__main__":
    main()
