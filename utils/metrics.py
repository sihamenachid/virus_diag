def _counts(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, fp, fn


def accuracy(y_true, y_pred):
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def precision(y_true, y_pred):
    tp, fp, _ = _counts(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y_true, y_pred):
    tp, _, fn = _counts(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r else 0.0
