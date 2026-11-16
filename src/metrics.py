import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from get_f1_score import calc_p2p, point_adjust

AUTO_RANGE_WINDOW_CAP = 250


# ==========================================
# Part 1: 基础工具函数 (Numpy 极速版)
# ==========================================

def convert_vector_to_events(vector: np.ndarray) -> List[Tuple[int, int]]:
    """
    将二进制向量转换为事件列表。
    返回格式: [(start, end), ... ] (exclusive end)
    """
    if len(vector) == 0:
        return []

    vector = np.asarray(vector)
    padded = np.concatenate([[0], vector, [0]])
    diffs = np.diff(padded)

    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # 确保返回列表而非 zip 迭代器
    return list(zip(starts.tolist(), ends.tolist()))


def range_convers_new(label: np.ndarray) -> List[Tuple[int, int]]:
    """
    原始代码的 segment 检测方式。
    返回 inclusive 的 (start, end) 区间。
    """
    label = np.asarray(label)
    if len(label) == 0:
        return []

    label_binary = (label != 0).astype(np.int8)
    padded = np.concatenate([[0], label_binary, [0]])
    diffs = np.diff(padded)

    starts = np.where(diffs == 1)[0]
    # inclusive end
    ends = np.where(diffs == -1)[0] - 1

    return list(zip(starts.tolist(), ends.tolist()))


def interval_intersection(I: Optional[Tuple], J: Optional[Tuple]) -> Optional[Tuple]:
    if I is None or J is None: return None
    # 展开 max/min 减少函数调用开销
    start = I[0] if I[0] > J[0] else J[0]
    end = I[1] if I[1] < J[1] else J[1]
    if start >= end: return None
    return (start, end)


def interval_length(I: Optional[Tuple]) -> float:
    return I[1] - I[0] if I is not None else 0


def sum_interval_lengths(intervals: List) -> float:
    return sum(interval_length(I) for I in intervals if I is not None)


# ==========================================
# Part 2, 3, 4: Affiliation Metrics
# (逻辑复杂且非主要瓶颈，保持逻辑精确性)
# ==========================================

def t_start(j: int, Js: List[Tuple], Trange: Tuple) -> float:
    if j == len(Js):
        return 2 * max(Trange) - t_stop(len(Js) - 1, Js, Trange)
    return Js[j][0]


def t_stop(j: int, Js: List[Tuple], Trange: Tuple) -> float:
    if j == -1:
        return 2 * min(Trange) - t_start(0, Js, Trange)
    return Js[j][1]


def E_gt_func(j: int, Js: List[Tuple], Trange: Tuple) -> Tuple[float, float]:
    return ((t_stop(j - 1, Js, Trange) + t_start(j, Js, Trange)) / 2,
            (t_stop(j, Js, Trange) + t_start(j + 1, Js, Trange)) / 2)


def get_all_E_gt_func(Js: List[Tuple], Trange: Tuple) -> List[Tuple[float, float]]:
    return [E_gt_func(j, Js, Trange) for j in range(len(Js))]


def affiliation_partition(Is: List[Tuple], E_gt: List[Tuple]) -> List[List]:
    out = []
    # 简单空间换时间优化：避免重复计算
    for E_gt_j in E_gt:
        zone_intervals = []
        for I in Is:
            # 快速过滤
            if I[1] <= E_gt_j[0] or I[0] >= E_gt_j[1]: continue

            # 展开 intersection 逻辑
            start = I[0] if I[0] > E_gt_j[0] else E_gt_j[0]
            end = I[1] if I[1] < E_gt_j[1] else E_gt_j[1]
            if start < end:
                zone_intervals.append((start, end))
        out.append(zone_intervals)
    return out


# ==========================================
# Part 3: Affiliation 积分计算 (保持原样)
# ==========================================

def cut_into_three_func(I: Optional[Tuple], J: Tuple) -> Tuple:
    if I is None: return (None, None, None)
    I_inter_J = interval_intersection(I, J)
    if I_inter_J is None:
        if I[1] <= J[0]:
            return (I, None, None)
        elif I[0] >= J[1]:
            return (None, None, I)
        else:
            return (None, None, None)
    I_before = (I[0], J[0]) if I[0] < J[0] else None
    I_after = (J[1], I[1]) if I[1] > J[1] else None
    return (I_before, I_inter_J, I_after)


def integral_mini_interval_P_CDFmethod__min_piece(I: Tuple, J: Tuple, E: Tuple) -> float:
    e_min, e_max = E[0], E[1]
    j_min, j_max = J[0], J[1]
    i_min, i_max = I[0], I[1]
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    m = min(j_min - e_min, e_max - j_max)
    A = min(d_max, m) ** 2 - min(d_min, m) ** 2
    B = max(d_max, m) - max(d_min, m)
    C = 0.5 * A + m * B
    return C


def integral_mini_interval_Pprecision_CDFmethod(I: Tuple, J: Tuple, E: Tuple) -> float:
    integral_min_piece = integral_mini_interval_P_CDFmethod__min_piece(I, J, E)
    e_min, e_max = E[0], E[1]
    j_min, j_max = J[0], J[1]
    i_min, i_max = I[0], I[1]
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    integral_linear_piece = 0.5 * (d_max ** 2 - d_min ** 2)
    integral_remaining_piece = (j_max - j_min) * (i_max - i_min)
    DeltaI = i_max - i_min
    DeltaE = e_max - e_min
    if DeltaE == 0: return 0
    return DeltaI - (1 / DeltaE) * (integral_min_piece + integral_linear_piece + integral_remaining_piece)


def integral_interval_probaCDF_precision(I: Optional[Tuple], J: Tuple, E: Tuple) -> float:
    if I is None: return 0
    cut = cut_into_three_func(I, J)
    d_left = integral_mini_interval_Pprecision_CDFmethod(cut[0], J, E) if cut[0] else 0
    d_middle = cut[1][1] - cut[1][0] if cut[1] else 0
    d_right = integral_mini_interval_Pprecision_CDFmethod(cut[2], J, E) if cut[2] else 0
    return d_left + d_middle + d_right


def get_pivot_j(I: Tuple, J: Tuple) -> float:
    if I[1] <= J[0]:
        return J[0]
    elif I[0] >= J[1]:
        return J[1]
    else:
        raise ValueError("I and J should not intersect")


def cut_J_based_on_mean_func(J: Optional[Tuple], e_mean: float) -> Tuple:
    if J is None: return (None, None)
    if e_mean >= J[1]:
        return (J, None)
    elif e_mean <= J[0]:
        return (None, J)
    else:
        return ((J[0], e_mean), (e_mean, J[1]))


def integral_mini_interval_Precall_CDFmethod(I: Tuple, J: Tuple, E: Tuple) -> float:
    try:
        i_pivot = get_pivot_j(J, I)
    except ValueError:
        return 0
    e_min, e_max = E[0], E[1]
    if i_pivot <= e_min or i_pivot >= e_max: return 0
    e_mean = (e_min + e_max) / 2
    J_before, J_after = cut_J_based_on_mean_func(J, e_mean)
    iemin_mean = (e_min + i_pivot) / 2
    J_before_closeE, J_before_closeI = cut_J_based_on_mean_func(J_before, iemin_mean)
    iemax_mean = (e_max + i_pivot) / 2
    J_after_closeI, J_after_closeE = cut_J_based_on_mean_func(J_after, iemax_mean)

    def get_bounds(interval):
        return (interval[0], interval[1]) if interval else (math.nan, math.nan)

    j_bb_min, j_bb_max = get_bounds(J_before_closeE)
    j_ba_min, j_ba_max = get_bounds(J_before_closeI)
    j_ab_min, j_ab_max = get_bounds(J_after_closeI)
    j_aa_min, j_aa_max = get_bounds(J_after_closeE)

    def safe_val(expr):
        return 0 if math.isnan(expr) else expr

    if i_pivot >= J[1]:
        p1 = safe_val((i_pivot - e_min) * (j_bb_max - j_bb_min))
        p2 = safe_val(2 * i_pivot * (j_ba_max - j_ba_min) - (j_ba_max ** 2 - j_ba_min ** 2))
        p3 = safe_val(2 * i_pivot * (j_ab_max - j_ab_min) - (j_ab_max ** 2 - j_ab_min ** 2))
        p4 = safe_val((e_max + i_pivot) * (j_aa_max - j_aa_min) - (j_aa_max ** 2 - j_aa_min ** 2))
    else:
        p1 = safe_val((j_bb_max ** 2 - j_bb_min ** 2) - (e_min + i_pivot) * (j_bb_max - j_bb_min))
        p2 = safe_val((j_ba_max ** 2 - j_ba_min ** 2) - 2 * i_pivot * (j_ba_max - j_ba_min))
        p3 = safe_val((j_ab_max ** 2 - j_ab_min ** 2) - 2 * i_pivot * (j_ab_max - j_ab_min))
        p4 = safe_val((e_max - i_pivot) * (j_aa_max - j_aa_min))

    out_integral = p1 + p2 + p3 + p4
    DeltaJ = J[1] - J[0]
    DeltaE = e_max - e_min
    return DeltaJ - (1 / DeltaE) * out_integral if DeltaE > 0 else 0


def integral_interval_probaCDF_recall(I: Optional[Tuple], J: Optional[Tuple], E: Tuple) -> float:
    if I is None or J is None: return 0
    cut = cut_into_three_func(J, I)
    d_left = integral_mini_interval_Precall_CDFmethod(I, cut[0], E) if cut[0] else 0
    d_middle = cut[1][1] - cut[1][0] if cut[1] else 0
    d_right = integral_mini_interval_Precall_CDFmethod(I, cut[2], E) if cut[2] else 0
    return d_left + d_middle + d_right


def affiliation_precision_proba(Is: List, J: Tuple, E: Tuple) -> float:
    Is_filtered = [I for I in Is if I is not None]
    if not Is_filtered: return math.nan
    total_length = sum_interval_lengths(Is_filtered)
    if total_length == 0: return math.nan
    total_integral = sum(integral_interval_probaCDF_precision(I, J, E) for I in Is_filtered)
    return total_integral / total_length


def affiliation_recall_proba(Is: List, J: Tuple, E: Tuple) -> float:
    Is_filtered = [I for I in Is if I is not None]
    if not Is_filtered: return 0
    j_length = interval_length(J)
    if j_length == 0: return 0
    E_gt_recall = get_all_E_gt_func(Is_filtered, E)
    Js_partitioned = affiliation_partition([J], E_gt_recall)
    total_integral = 0
    for i, I in enumerate(Is_filtered):
        for J_part in Js_partitioned[i]:
            if J_part is not None:
                total_integral += integral_interval_probaCDF_recall(I, J_part, E)
    return total_integral / j_length


def pr_from_events(events_pred: List, events_gt: List, Trange: Tuple) -> Dict:
    if not events_gt:
        return {"Affiliation_Precision": math.nan, "Affiliation_Recall": math.nan}
    E_gt = get_all_E_gt_func(events_gt, Trange)
    Is_partitioned = affiliation_partition(events_pred, E_gt)
    precisions = []
    recalls = []
    for j, J in enumerate(events_gt):
        prec = affiliation_precision_proba(Is_partitioned[j], events_gt[j], E_gt[j])
        rec = affiliation_recall_proba(Is_partitioned[j], events_gt[j], E_gt[j])
        precisions.append(prec)
        recalls.append(rec)
    valid_precs = [p for p in precisions if not math.isnan(p)]
    valid_recs = [r for r in recalls if not math.isnan(r)]
    avg_precision = sum(valid_precs) / len(valid_precs) if valid_precs else 0
    avg_recall = sum(valid_recs) / len(valid_recs) if valid_recs else 0
    return {"Affiliation_Precision": avg_precision, "Affiliation_Recall": avg_recall}


# ==========================================
# Part 5: Range-AUC 终极优化版
# ==========================================

def extend_positive_range(labels: np.ndarray, window: int) -> np.ndarray:
    """
    Extend anomaly ranges with the symmetric VUS buffer rule.

    Range ends are inclusive.  The right buffer therefore starts at
    ``e + 1`` and includes the outermost point.  Overlapping buffers are
    accumulated and clipped, matching the reference range-AUC procedure.
    """
    if window <= 0:
        return labels.astype(np.float64)

    label = labels.astype(np.float64)
    length = len(label)

    L = range_convers_new(labels)
    if not L:
        return label

    for s, e in L:
        right_start = e + 1
        right_end = min(e + window // 2 + 1, length)
        if right_end > right_start:
            x = np.arange(right_start, right_end)
            vals = np.sqrt(1 - (x - e) / window)
            label[right_start:right_end] += vals

        start_b = max(s - window // 2, 0)
        if start_b < s:
            x = np.arange(start_b, s)
            vals = np.sqrt(1 - (s - x) / window)
            label[start_b:s] += vals

    np.minimum(label, 1.0, out=label)

    return label


def _prepare_range_auc_score(
    score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or score.size == 0 or not np.isfinite(score).all():
        raise ValueError("Range-AUC scores must be a finite non-empty vector")
    unique_scores = np.unique(score)[::-1]
    n_points = min(250, len(unique_scores))
    indices = np.rint(
        np.linspace(0, len(unique_scores) - 1, n_points)
    ).astype(np.int64)
    thresholds = np.unique(unique_scores[indices])[::-1]
    return score, thresholds


def _range_auc_from_prepared_score(
    labels_original: np.ndarray,
    prepared_score: tuple[
        np.ndarray,
        np.ndarray,
    ],
    window: int,
) -> Tuple[float, float]:
    score, thresholds = prepared_score
    labels_extended = extend_positive_range(labels_original, window)

    P = float(np.sum(labels_original))
    P_new = (P + np.sum(labels_extended)) / 2

    L = range_convers_new(labels_extended)
    if not L:
        return 0.0, 0.0

    n = len(score)
    N_new = n - P_new
    num_segments = len(L)

    # Compute threshold predictions with actual tie-aware counts.  Sorting
    # once and using searchsorted preserves the semantics of score >=
    # threshold without allocating one full boolean mask per threshold.
    order = np.argsort(score, kind="stable")[::-1]
    sorted_scores = score[order]
    sorted_labels = labels_extended[order]
    cumulative_tp = np.cumsum(sorted_labels)
    counts = np.searchsorted(
        -sorted_scores,
        -thresholds,
        side="right",
    ).astype(np.int64)
    true_positives = np.where(
        counts > 0,
        cumulative_tp[np.maximum(counts - 1, 0)],
        0.0,
    )
    predicted_counts = counts.astype(np.float64)
    false_positives = predicted_counts - true_positives
    recalls = (
        np.minimum(true_positives / P_new, 1.0)
        if P_new > 0
        else np.zeros_like(true_positives)
    )
    segment_max_scores = np.asarray(
        [np.max(score[s:e + 1]) for s, e in L],
        dtype=np.float64,
    )
    existence = (
        segment_max_scores[:, None] >= thresholds[None, :]
    ).mean(axis=0)
    tprs = recalls * existence
    fprs = (
        false_positives / N_new
        if N_new > 0
        else np.zeros_like(false_positives)
    )
    precisions = np.divide(
        true_positives,
        predicted_counts,
        out=np.ones_like(true_positives),
        where=predicted_counts > 0,
    )

    tpr_roc = np.concatenate(([0.0], tprs, [1.0]))
    fpr_roc = np.concatenate(([0.0], fprs, [1.0]))
    AUC_ROC = float(np.trapezoid(tpr_roc, fpr_roc))

    tpr_pr = np.concatenate(([0.0], tprs, [1.0]))
    prec_pr = np.concatenate(([1.0], precisions, [1.0]))
    AUC_PR = float(np.trapezoid(prec_pr, tpr_pr))

    return AUC_ROC, AUC_PR


def RangeAUC_vectorized(labels_original: np.ndarray, score: np.ndarray, window: int) -> Tuple[float, float]:
    """
    [终极优化] Range-AUC 完全矩阵化版本 (Bit-exact)
    """
    return _range_auc_from_prepared_score(
        labels_original,
        _prepare_range_auc_score(score),
        window,
    )


def RangeAUC_volume(labels_original: np.ndarray, score: np.ndarray,
                    windowSize: int) -> Tuple[float, float]:
    """
    计算 VUS
    保留 step=1，确保积分精度与原代码完全一致
    """
    auc_rocs = []
    auc_prs = []
    prepared_score = _prepare_range_auc_score(score)

    for w in range(windowSize + 1):
        auc_roc, auc_pr = _range_auc_from_prepared_score(
            labels_original,
            prepared_score,
            w,
        )
        auc_rocs.append(auc_roc)
        auc_prs.append(auc_pr)

    n = len(auc_rocs)
    return sum(auc_rocs) / n, sum(auc_prs) / n


def automatic_range_window(labels: np.ndarray) -> int:
    """Resolve the reproducible automatic buffer used by formal metrics."""
    events = convert_vector_to_events(np.asarray(labels, dtype=np.int32))
    lengths = [end - start for start, end in events]
    median_length = int(np.median(lengths)) if lengths else 100
    return min(max(median_length, 1), AUTO_RANGE_WINDOW_CAP)


# ==========================================
# Part 6: 入口
# ==========================================

def evaluate_metrics(y_score: np.ndarray, y_true: np.ndarray,
                     slidingWindow: Optional[int] = None,
                     affiliation_threshold: Optional[float] = None) -> Dict[str, float]:
    y_score = np.asarray(y_score, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    if slidingWindow is None:
        slidingWindow = automatic_range_window(y_true)

    # Affiliation
    if affiliation_threshold is None:
        thresholds = np.unique(y_score)
        if len(thresholds) > 100:
            thresholds = np.percentile(y_score, np.linspace(0, 100, 100))
    else:
        if not np.isfinite(affiliation_threshold):
            raise ValueError("affiliation_threshold must be finite")
        thresholds = np.asarray([affiliation_threshold], dtype=np.float64)

    events_gt = convert_vector_to_events(y_true)
    events_gt = list(events_gt) if not isinstance(events_gt, list) else events_gt
    Trange = (0, len(y_true))

    best_aff_f1 = 0
    best_aff_prec = 0
    best_aff_rec = 0

    for th in thresholds:
        pred_binary = (y_score >= th).astype(int)
        events_pred = convert_vector_to_events(pred_binary)

        if not events_pred:
            prec, rec = 0.0, 0.0
        else:
            result = pr_from_events(events_pred, events_gt, Trange)
            prec = result["Affiliation_Precision"]
            rec = result["Affiliation_Recall"]

        if math.isnan(prec): prec = 0
        if math.isnan(rec): rec = 0
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        if f1 > best_aff_f1:
            best_aff_f1 = f1
            best_aff_prec = prec
            best_aff_rec = rec

    # Range-AUC
    r_auc_roc, r_auc_pr = RangeAUC_vectorized(y_true, y_score, slidingWindow)

    # VUS (耗时大户，已优化)
    vus_roc, vus_pr = RangeAUC_volume(y_true, y_score, 2 * slidingWindow)

    return {
        "affiliation_precision": best_aff_prec,
        "affiliation_recall": best_aff_rec,
        "affiliation_f1": best_aff_f1,
        "r_auc_roc": r_auc_roc,
        "r_auc_pr": r_auc_pr,
        "vus_roc": vus_roc,
        "vus_pr": vus_pr
    }


FORMAL_METRICS = (
    "P", "R", "F1",
    "P_PA", "R_PA", "F1_PA",
    "Aff-F1",
    "AUC-ROC", "AUC-PR",
    "R-AUC-ROC", "R-AUC-PR",
    "VUS-ROC", "VUS-PR",
)


def compute_all_metrics(score: np.ndarray, label: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute the fixed 13 metrics using one predeclared training threshold."""
    scores = np.asarray(score, dtype=np.float64).ravel()
    labels = np.asarray(label, dtype=np.int32).ravel()
    if scores.size == 0 or scores.shape != labels.shape:
        raise ValueError("Scores and labels must be non-empty arrays of equal length")
    if not np.isfinite(scores).all():
        raise FloatingPointError("Scores contain NaN/Inf values")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary")
    if np.unique(labels).size != 2:
        raise ValueError("Formal AUC metrics require both normal and anomalous labels")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    range_metrics = evaluate_metrics(scores, labels, affiliation_threshold=float(threshold))
    raw_prediction = scores >= threshold
    f1_raw, precision_raw, recall_raw, *_ = calc_p2p(raw_prediction, labels)
    adjusted_prediction, actual = point_adjust(scores, labels, float(threshold))
    f1_pa, precision_pa, recall_pa, *_ = calc_p2p(adjusted_prediction, actual)
    precision_curve, recall_curve, _ = precision_recall_curve(
        labels,
        scores,
    )
    result = {
        "P": float(precision_raw),
        "R": float(recall_raw),
        "F1": float(f1_raw),
        "P_PA": float(precision_pa),
        "R_PA": float(recall_pa),
        "F1_PA": float(f1_pa),
        "Aff-F1": float(range_metrics["affiliation_f1"]),
        "AUC-ROC": float(roc_auc_score(labels, scores)),
        "AUC-PR": float(
            auc(
                recall_curve[::-1],
                precision_curve[::-1],
            )
        ),
        "R-AUC-ROC": float(range_metrics["r_auc_roc"]),
        "R-AUC-PR": float(range_metrics["r_auc_pr"]),
        "VUS-ROC": float(range_metrics["vus_roc"]),
        "VUS-PR": float(range_metrics["vus_pr"]),
    }
    if tuple(result) != FORMAL_METRICS:
        raise RuntimeError("Internal formal metric ordering contract was violated")
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Formal metric computation produced NaN/Inf")
    return result
