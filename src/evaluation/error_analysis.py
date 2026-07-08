import numpy as np
from src.evaluation.seqeval_metrics import seqeval_strict_micro_f1
from src.evaluation.spans import bio_to_spans

def boundary_error_breakdown(y_true, y_pred):
    stats = {
        "exact": 0,
        "type_confusion": 0,
        "boundary_mismatch": 0,
        "missing": 0,
        "spurious": 0,
        "total_gold": 0,
        "total_pred": 0,
    }

    def overlaps(a, b):
        s1, e1, _ = a
        s2, e2, _ = b
        return max(s1, s2) < min(e1, e2)

    for gt, pr in zip(y_true, y_pred):
        gsp = bio_to_spans(gt)
        psp = bio_to_spans(pr)
        stats["total_gold"] += len(gsp)
        stats["total_pred"] += len(psp)

        gset = set(gsp)
        pset = set(psp)
        exact = gset & pset
        stats["exact"] += len(exact)

        gb = {(s, e): t for s, e, t in gsp}
        pb = {(s, e): t for s, e, t in psp}

        for b in set(gb.keys()) & set(pb.keys()):
            if gb[b] != pb[b]:
                stats["type_confusion"] += 1

        for g in gsp:
            if g in exact:
                continue
            if any(overlaps(g, p) for p in psp):
                stats["boundary_mismatch"] += 1
            else:
                stats["missing"] += 1

        for p in psp:
            if p in exact:
                continue
            if any(overlaps(p, g) for g in gsp):
                continue
            stats["spurious"] += 1

    return stats

def summarize(model_name, y_true, y_pred):
    res = seqeval_strict_micro_f1(y_true, y_pred)
    b = boundary_error_breakdown(y_true, y_pred)
    return {
        "model": model_name,
        "f1": res["overall_f1"],
        "precision": res["overall_precision"],
        "recall": res["overall_recall"],
        "exact_spans": b["exact"],
        "type_confusion": b["type_confusion"],
        "boundary_mismatch": b["boundary_mismatch"],
        "missing": b["missing"],
        "spurious": b["spurious"],
        "gold_spans": b["total_gold"],
        "pred_spans": b["total_pred"],
    }
