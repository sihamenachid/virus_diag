from utils.metrics import accuracy, precision, recall, f1_score


def test_accuracy_parfaite():
    assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0


def test_accuracy_moitie():
    assert accuracy([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5


def test_precision_recall_f1():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    assert precision(y_true, y_pred) == 0.5
    assert recall(y_true, y_pred) == 0.5
    assert f1_score(y_true, y_pred) == 0.5


def test_pas_de_division_par_zero():
    assert precision([0, 0], [0, 0]) == 0.0
    assert recall([0, 0], [0, 0]) == 0.0
