import re
from typing import List, Tuple

def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokens_to_text_and_offsets(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    parts, offsets = [], []
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

def bio_to_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
    spans = []
    start, ent_type = None, None
    for i, lab in enumerate(labels):
        if lab == "O":
            if ent_type is not None:
                spans.append((start, i, ent_type))
                start, ent_type = None, None
            continue
        prefix, typ = lab.split("-", 1)
        if prefix == "B":
            if ent_type is not None:
                spans.append((start, i, ent_type))
            start, ent_type = i, typ
        elif prefix == "I":
            if ent_type is None:
                start, ent_type = i, typ
            elif typ != ent_type:
                spans.append((start, i, ent_type))
                start, ent_type = i, typ
    if ent_type is not None:
        spans.append((start, len(labels), ent_type))
    return spans

def spans_char_from_bio(tokens, offsets, labels):
    spans = bio_to_spans(labels)
    char_spans = []
    for start, end, ent_type in spans:
        cs = offsets[start][0]
        ce = offsets[end - 1][1]
        char_spans.append((cs, ce, ent_type))
    return char_spans

def char_spans_to_bio(tokens, offsets, char_spans, label_set):
    labels = ["O"] * len(tokens)
    char_spans = sorted(char_spans, key=lambda x: (x[0], -(x[1] - x[0])))
    for cs, ce, ent_type in char_spans:
        if ent_type not in label_set:
            continue
        covered = []
        for i, (ts, te) in enumerate(offsets):
            if ts >= cs and te <= ce:
                covered.append(i)
        if not covered:
            continue
        labels[covered[0]] = "B-" + ent_type
        for idx in covered[1:]:
            labels[idx] = "I-" + ent_type
    return labels
