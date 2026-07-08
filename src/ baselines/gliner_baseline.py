import torch
from gliner import GLiNER

from src.data.prepare_dataset import tokens_to_text_and_offsets
from src.evaluation.spans import char_spans_to_bio
from src.data.prepare_labels import ENTITY_TYPES


def load_gliner(model_id="urchade/gliner_multi-v2.1", device="cpu"):
    """
    Load GLiNER model.
    """
    return GLiNER.from_pretrained(model_id).to(device)


def gliner_char_spans(text: str, gliner_model, threshold=0.35):
    """
    Run GLiNER and return character-level spans mapped to PER/ORG/LOC.
    """
    ents = gliner_model.predict_entities(
        text,
        labels=["person", "organization", "location"],
        threshold=threshold
    )

    out = []
    for ent in ents:
        lab = ent["label"].lower()

        if lab.startswith("person"):
            ent_type = "PER"
        elif lab.startswith("organization"):
            ent_type = "ORG"
        elif lab.startswith("location"):
            ent_type = "LOC"
        else:
            continue

        out.append((ent["start"], ent["end"], ent_type))

    return out


def run_gliner_on_dataset(ds_split, gliner_model, factru_example_to_labels, limit=None, threshold=0.35):
    """
    Convert GLiNER spans to BIO labels for a dataset split.
    """
    y_true, y_pred = [], []
    n = len(ds_split) if limit is None else min(limit, len(ds_split))

    for i in range(n):
        example = ds_split[i]

        tokens, gold = factru_example_to_labels(example)
        text, offsets = tokens_to_text_and_offsets(tokens)

        spans = gliner_char_spans(text, gliner_model, threshold=threshold)
        pred = char_spans_to_bio(tokens, offsets, spans, label_set=ENTITY_TYPES)

        y_true.append(gold)
        y_pred.append(pred)
    return y_true, y_pred

