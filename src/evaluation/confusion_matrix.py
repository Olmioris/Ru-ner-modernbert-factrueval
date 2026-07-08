import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.evaluation.spans import bio_to_spans

def confusion_matrix_by_type(y_true, y_pred, entity_types):
    labels = list(entity_types)
    idx = {t: i for i, t in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for gt, pr in zip(y_true, y_pred):
        gsp = bio_to_spans(gt)
        psp = bio_to_spans(pr)
        gb = {(s, e): t for s, e, t in gsp}
        pb = {(s, e): t for s, e, t in psp}
        for b in set(gb.keys()) & set(pb.keys()):
            g = gb[b]
            p = pb[b]
            if g in idx and p in idx:
                cm[idx[g], idx[p]] += 1
    return cm, labels

def plot_confusion(cm, labels, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("gold")
    plt.show()
