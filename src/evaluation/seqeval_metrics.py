import evaluate

seqeval_metric = evaluate.load("seqeval")

def seqeval_strict_micro_f1(y_true, y_pred):
    """
    Compute strict micro F1 using seqeval.
    """
    return seqeval_metric.compute(predictions=y_pred, references=y_true, zero_division=0)
