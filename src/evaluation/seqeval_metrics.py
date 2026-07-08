from seqeval.metrics import precision_score, recall_score, f1_score

def compute_seqeval_metrics(y_true, y_pred):
    """
    Compute precision, recall, and F1 using seqeval.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1
