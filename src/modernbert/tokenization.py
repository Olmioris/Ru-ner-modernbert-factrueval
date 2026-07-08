from transformers import AutoTokenizer

def load_tokenizer(model_id="deepvk/RuModernBERT-base"):
    return AutoTokenizer.from_pretrained(model_id)
