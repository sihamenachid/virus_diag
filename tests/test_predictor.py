from app.interface_clinique import ClinicalPredictor


class FakeModel:
    """Faux modèle qui renvoie toujours la même probabilité."""

    def __init__(self, probability):
        self.probability = probability

    def predict(self, x):
        return self.probability


def test_infecte_si_prediction_superieure_ou_egale_a_05():
    assert ClinicalPredictor(FakeModel(0.5)).diagnose([0, 0, 0, 0]) == "Infecté"
    assert ClinicalPredictor(FakeModel(0.9)).diagnose([0, 0, 0, 0]) == "Infecté"


def test_sain_si_prediction_inferieure_a_05():
    assert ClinicalPredictor(FakeModel(0.2)).diagnose([0, 0, 0, 0]) == "Sain"
