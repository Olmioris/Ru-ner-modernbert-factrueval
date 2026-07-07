def build_label_maps(ds):
    all_labels = set()
    for ex in ds["train"]:
        all_labels.update(ex["ner_tags_str"])

    other_labels = sorted(l for l in all_labels if l != "O")
    label_names = ["O"] + other_labels

    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in enumerate(label_names)}
    ENTITY_TYPES = ("PER", "ORG", "LOC")

    return label_names, id2label, label2id, ENTITY_TYPES


def add_numeric_labels(example, label2id):
    example["ner_tags"] = [label2id[tag] for tag in example["ner_tags_str"]]
    return example
