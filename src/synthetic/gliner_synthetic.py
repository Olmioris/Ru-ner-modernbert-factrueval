import torch
from datasets import Dataset, DatasetDict

from src.data.prepare_dataset import tokens_to_text_and_offsets
from src.evaluation.spans import char_spans_to_bio
from src.data.prepare_labels import ENTITY_TYPES
from gliner import GLiNER


def load_gliner(model_id="urchade/gliner_multi-v2.1", device="cpu"):
    """
    Load GLiNER model for synthetic NER generation.
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


def generate_synthetic_labels(text_ds, gliner_model, label_set=ENTITY_TYPES):
    """
    Generate synthetic BIO labels for a dataset of reconstructed text.
    """
    synthetic_samples = []

    for example in text_ds:
        text = example["text"]
        offsets = example["offsets"]

        spans = gliner_char_spans(text, gliner_model)
        tokens = text.split()

        bio_labels = char_spans_to_bio(tokens, offsets, spans, label_set=label_set)

        synthetic_samples.append(
            {
                "tokens": tokens,
                "ner_tags_str": bio_labels,
                "length": len(tokens)
            }
        )

    synthetic_ds = Dataset.from_list(synthetic_samples)
    return DatasetDict({"train": synthetic_ds})

