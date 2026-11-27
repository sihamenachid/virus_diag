import pytest

from core.dataset import Dataset
from core.model import Model
from core.optimizer import GradientDescent
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.svm import SVM
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator


def test_model_est_abstrait():
    with pytest.raises(TypeError):
        Model()


@pytest.mark.parametrize("model", [
    LogisticRegression(n_features=4),
    NeuralNetwork(n_features=4),
    SVM(n_features=4),
])
def test_predict_retourne_une_probabilite(model):
    p = model.predict([1.0, 1, 0, 1])
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("model, lr, epochs", [
    (LogisticRegression(n_features=4), 0.5, 200),
    (NeuralNetwork(n_features=4, n_hidden=4), 0.05, 100),
    (SVM(n_features=4), 0.1, 200),
])
def test_accuracy_superieure_a_80(model, lr, epochs):
    X_train, X_test, y_train, y_test = Dataset("data/patient_data.csv").split()
    Trainer(model, GradientDescent(lr), epochs).fit(X_train, y_train, verbose=False)
    scores = Evaluator(model).evaluate(X_test, y_test)
    assert scores["accuracy"] >= 0.80
