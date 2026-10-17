import json

from core.dataset import Dataset
from app.interface_clinique import ClinicalPredictor


class DiagnosisAPI:
    """API pour applications externes : reçoit du JSON, renvoie du JSON."""

    def __init__(self, predictor: ClinicalPredictor):
        self.predictor = predictor

    def handle_request(self, json_request):
        data = json.loads(json_request)
        patient = [float(data[f]) for f in Dataset.FEATURES]
        return json.dumps({"patient": data, "diagnostic": self.predictor.diagnose(patient)},
                          ensure_ascii=False)
