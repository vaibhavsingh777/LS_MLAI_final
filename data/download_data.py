# data/download_data.py

from datasets import load_dataset

def filter_empty(example):
    """Remove empty or whitespace-only text samples."""
    return example['text'].strip() != ''

def main():
    # Load the WikiText-2 dataset (raw version)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    print("Original dataset splits and sizes:")
    print(dataset)

    # Filter out empty samples in each split
    filtered_dataset = dataset.filter(filter_empty)
    print("\nFiltered dataset splits and sizes:")
    print(filtered_dataset)

    # Show a non-empty sample from each split
    for split in ['train', 'validation', 'test']:
        for sample in filtered_dataset[split]:
            if sample['text'].strip():
                print(f"\n--- {split.upper()} SAMPLE ---")
                print(sample)
                break

    # Save filtered dataset for the next step
    filtered_dataset.save_to_disk("./data/filtered_wikitext2")
    print("\nFiltered dataset saved to ./data/filtered_wikitext2")

if __name__ == "__main__":
    main()
