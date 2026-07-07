from datasets import Dataset, DatasetDict, load_dataset

def unwrap_split(split_ds: Dataset) -> Dataset:
    data_list = split_ds[0]["data"]
    return Dataset.from_list(data_list)

def load_factrueval():
    raw = load_dataset("gusevski/factrueval2016")
    ds = DatasetDict(
        {
            "train": unwrap_split(raw["train"]),
            "validation": unwrap_split(raw["validation"]),
            "test": unwrap_split(raw["test"]),
        }
    )
    return ds
