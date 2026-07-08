from datasets import DatasetDict

def merge_original_and_synthetic(original_ds, synthetic_ds):
    """
    Merge original and synthetic datasets for augmented training.
    """
    return DatasetDict(
        {
            "train": original_ds["train"] + synthetic_ds["train"],
            "validation": original_ds["validation"],
            "test": original_ds["test"]
        }
    )
