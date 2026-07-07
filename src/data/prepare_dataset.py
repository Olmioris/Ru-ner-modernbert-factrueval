import re
from typing import List, Tuple
from datasets import DatasetDict

def create_small_splits(ds, train_size, val_size, test_size, seed):
    ds_small = DatasetDict(
        {
            "train": ds["train"].shuffle(seed=seed).select(range(train_size)),
            "validation": ds["validation"].shuffle(seed=seed).select(range(val_size)),
            "test": ds["test"].shuffle(seed=seed).select(range(test_size)),
        }
    )
    return ds_small


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens_to_text_and_offsets(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    parts = []
    offsets = []
    cur = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
            cur += 1
        start = cur
        parts.append(tok)
        cur += len(tok)
        end = cur
        offsets.append((start, end))
    text = normalize_text("".join(parts))
    return text, offsets

