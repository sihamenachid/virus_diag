from core.dataset import Dataset
from core.optimizer import GradientDescent
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from app.interface_clinique import ClinicalPredictor

# 1. Données
dataset = Dataset("data/patient_data.csv")
X_train, X_test, y_train, y_test = dataset.split(test_ratio=0.2)
print(f"Patients : {len(dataset)} | entraînement : {len(X_train)} | test : {len(X_test)}")

# Nouveau patient : [ecart_temperature (°C au-dessus de 37), fievre, toux, fatigue]
# ex : 39.2 °C -> 2.2
patient = [2.2, 1, 1, 0]

# 2. Deux modèles différents, même pipeline (polymorphisme)
models = {
    "LogisticRegression": (LogisticRegression(n_features=4), GradientDescent(0.5), 200),
    "NeuralNetwork": (NeuralNetwork(n_features=4, n_hidden=4), GradientDescent(0.05), 100),
}

for name, (model, optimizer, epochs) in models.items():
    print(f"\n=== {name} ===")
    Trainer(model, optimizer, epochs).fit(X_train, y_train)          # entraînement
    scores = Evaluator(model).evaluate(X_test, y_test)               # évaluation
    print("   " + " | ".join(f"{k} = {v:.2f}" for k, v in scores.items()))
    predictor = ClinicalPredictor(model)                             # prédiction
    print(f"   Diagnostic du patient {patient} : {predictor.diagnose(patient)}")
