"""Verification metrics: ROC, EER and GAR at fixed FAR."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import config


def roc(genuine, impostor):
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    s = np.r_[genuine, impostor]
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr


def equal_error_rate(fpr, tpr):
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def gar_at_far(fpr, tpr, far):
    """Highest GAR whose FAR does not exceed `far`."""
    i = np.searchsorted(fpr, far, side="right") - 1
    return float(tpr[max(i, 0)])


def summarize(genuine, impostor, fars=config.FAR_POINTS):
    fpr, tpr, _ = roc(genuine, impostor)
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    out = {
        "n_genuine": int(len(genuine)),
        "n_impostor": int(len(impostor)),
        "min_resolvable_far": 1.0 / max(len(impostor), 1),
        "eer": equal_error_rate(fpr, tpr),
        "auc": float(roc_auc_score(y, np.r_[genuine, impostor])),
    }
    for far in fars:
        out[f"gar@far={far:g}"] = gar_at_far(fpr, tpr, far)
    return out


def plot_rocs(curves, path, title=""):
    plt.figure(figsize=(6, 5))
    for name, (genuine, impostor) in curves.items():
        fpr, tpr, _ = roc(genuine, impostor)
        eer = equal_error_rate(fpr, tpr)
        plt.semilogx(np.maximum(fpr, 1e-6), tpr, label=f"{name} (EER {eer * 100:.2f}%)")
    plt.xlabel("False Accept Rate")
    plt.ylabel("Genuine Accept Rate")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
