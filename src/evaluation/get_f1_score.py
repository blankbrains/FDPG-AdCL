"""Point-wise and point-adjusted best-F1 utilities."""

from __future__ import annotations

import numpy as np


def calc_p2p(predict: np.ndarray, actual: np.ndarray):
    prediction = np.asarray(predict, dtype=bool)
    truth = np.asarray(actual, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and labels must have the same shape")
    tp = int(np.sum(prediction & truth))
    tn = int(np.sum(~prediction & ~truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return f1, precision, recall, tp, tn, fp, fn


def point_adjust(score: np.ndarray, label: np.ndarray, threshold: float):
    scores = np.asarray(score, dtype=np.float64).ravel()
    actual = np.asarray(label, dtype=np.int32).ravel().astype(bool)
    predict = scores >= threshold
    if scores.shape != actual.shape:
        raise ValueError("Scores and labels must have the same shape")
    in_anomaly = False
    for index in range(len(scores)):
        if actual[index] and predict[index] and not in_anomaly:
            in_anomaly = True
            cursor = index
            while cursor >= 0 and actual[cursor]:
                predict[cursor] = True
                cursor -= 1
        elif not actual[index]:
            in_anomaly = False
        if in_anomaly:
            predict[index] = True
    return predict, actual
