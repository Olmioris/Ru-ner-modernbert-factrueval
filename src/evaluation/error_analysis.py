from src.evaluation.spans import bio_to_spans

def summarize(model_name, y_true, y_pred):
    """
    Summarize evaluation results for one model.
    Returns a dictionary with metrics and error counts.
    """
    precision, recall, f1 = compute_seqeval_metrics(y_true, y_pred)

    gold_spans = sum(len(bio_to_spans(seq)) for seq in y_true)
    pred_spans = sum(len(bio_to_spans(seq)) for seq in y_pred)

    exact_spans = 0
    type_confusion = 0
    boundary_mismatch = 0
    missing = 0
    spurious = 0

    for true_seq, pred_seq in zip(y_true, y_pred):
        true_spans = bio_to_spans(true_seq)
        pred_spans = bio_to_spans(pred_seq)

        for span in true_spans:
            if span in pred_spans:
                exact_spans += 1
            else:
                # Check if same boundaries but wrong type
                same_boundary = any(
                    (span[0] == p[0] and span[1] == p[1]) for p in pred_spans
                )
                if same_boundary:
                    type_confusion += 1
                else:
                    # Check if overlapping boundaries
                    overlap = any(
                        (span[0] < p[1] and span[1] > p[0]) for p in pred_spans
                    )
                    if overlap:
                        boundary_mismatch += 1
                    else:
                        missing += 1

        for p in pred_spans:
            if p not in true_spans:
                spurious += 1

    return {
        "model": model_name,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "exact_spans": exact_spans,
        "type_confusion": type_confusion,
        "boundary_mismatch": boundary_mismatch,
        "missing": missing,
        "spurious": spurious,
        "gold_spans": gold_spans,
        "pred_spans": pred_spans,
    }


def collect_errors(y_true, y_pred, dataset):
    """
    Collect token-level errors for qualitative analysis.
    """
    errors = []
    for i, (true_seq, pred_seq) in enumerate(zip(y_true, y_pred)):
        tokens = dataset[i]["tokens"]
        for t, p, tok in zip(true_seq, pred_seq, tokens):
            if t != p:
                errors.append({
                    "token": tok,
                    "true": t,
                    "pred": p,
                    "index": i
                })
    return errors


def show_errors(errors, n=20):
    """
    Print first N errors.
    """
    for e in errors[:n]:
        print(f"Token: {e['token']}, True: {e['true']}, Pred: {e['pred']}, Example: {e['index']}")

