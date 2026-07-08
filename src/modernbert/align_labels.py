def align_labels(example, tokenizer, max_length=192):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length
    )

    labels = []
    word_ids = tokenized.word_ids()

    for i, w in enumerate(word_ids):
        if w is None:
            labels.append(-100)
        else:
            if i == 0 or w != word_ids[i - 1]:
                labels.append(example["ner_tags"][w])
            else:
                labels.append(-100)

    tokenized["labels"] = labels
    return tokenized
