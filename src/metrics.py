import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

from annotations import N_LABELS, ID_TO_LABEL


def divide(a: int, b: int) -> int:
    """
    Divide a by b, return 0 if b is 0.
    :param a:
    :param b:
    :return:
    """
    return a / b if b > 0 else 0


def compute_metrics(p):
    """
    Compute the metrics for the multilabel classification task.
    :param p: 2 numpy arrays: predictions and true_labels
    :return: metrics (dict): f1 score on
    """
    # (1)
    predictions, true_labels = p

    # (2)
    predicted_labels = np.where(predictions > 0, np.ones(predictions.shape), np.zeros(predictions.shape))
    metrics = {}

    # (3)
    cm = multilabel_confusion_matrix(true_labels.reshape(-1, N_LABELS), predicted_labels.reshape(-1, N_LABELS))

    # (4)
    for label_idx, matrix in enumerate(cm):
        if label_idx == 0:
            continue  # We don't care about the label "O"
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        metrics[f"f1_{ID_TO_LABEL[label_idx]}"] = f1

    # (5)
    macro_f1 = sum(list(metrics.values())) / (N_LABELS - 1)
    metrics["macro_f1"] = macro_f1

    return metrics
