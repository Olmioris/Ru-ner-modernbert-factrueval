import torch
from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter

from src.data.prepare_dataset import tokens_to_text_and_offsets
from src.evaluation.spans import char_spans_to_bio
from src.data.prepare_labels import ENTITY_TYPES


segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)


def natasha_ner_char_spans(text: str):
    """
    Run Natasha NER and return character-level spans.
    """
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    spans = []
    for span in doc.spans:
        spans.append((span.start, span.stop, span.type))
    return spans


def run_natasha_on_dataset(ds_split, factru_example_to_labels, limit=None):
    """
    Convert Natasha spans to BIO labels for a dataset split.
    """
    y_true, y_pred = [], []
    n = len(ds_split) if limit is None else min(limit, len(ds_split))

    for i in range(n):
        example = ds_split[i]

        tokens, gold = factru_example_to_labels(example)
        text, offsets = tokens_to_text_and_offsets(tokens)

        spans = natasha_ner_char_spans(text)
        pred = char_spans_to_bio(tokens, offsets, spans, label_set=ENTITY_TYPES)

        y_true.append(gold)
        y_pred.append(pred)
    return y_true, y_pred

