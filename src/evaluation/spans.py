def bio_to_spans(bio_tags):
    """
    Convert BIO tags to character-level spans.
    Returns list of tuples (start, end, type).
    """
    spans = []
    start, end, ent_type = None, None, None

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if ent_type is not None:
                spans.append((start, end, ent_type))
            start = i
            end = i + 1
            ent_type = tag.split("-")[1]
        elif tag.startswith("I-") and ent_type == tag.split("-")[1]:
            end = i + 1
        else:
            if ent_type is not None:
                spans.append((start, end, ent_type))
                ent_type = None

    if ent_type is not None:
        spans.append((start, end, ent_type))

    return spans


def char_spans_to_bio(tokens, offsets, spans, label_set):
    """
    Convert character-level spans to BIO tags.
    """
    tags = ["O"] * len(tokens)

    for start, end, ent_type in spans:
        if ent_type not in label_set:
            continue
        for i, (s, e) in enumerate(offsets):
            if s >= start and e <= end:
                prefix = "B-" if tags[i] == "O" else "I-"
                tags[i] = prefix + ent_type

    return tags

