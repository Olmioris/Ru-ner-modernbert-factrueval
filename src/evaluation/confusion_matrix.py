import pandas as pd

def entity_confusion_matrix(y_true, y_pred, entity_types):
    """
    Compute confusion matrix for entity types (PER, ORG, LOC).
    """
    matrix = {t: {t2: 0 for t2 in entity_types} for t in entity_types}

    for true_seq, pred_seq in zip(y_true, y_pred):
        for t, p in zip(true_seq, pred_seq):
            if t != "O" and p != "O":
                t_type = t.split("-")[-1]
                p_type = p.split("-")[-1]
                if t_type in entity_types and p_type in entity_types:
                    matrix[t_type][p_type] += 1

    return pd.DataFrame(matrix).T
