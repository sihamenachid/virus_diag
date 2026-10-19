```
virus_diag/
├── data/patient_data.csv          données patients déjà préparées (300)
├── core/
│   ├── dataset.py                 chargement + séparation train/test
│   ├── model.py                   classe abstraite Model (abstraction)
│   ├── logistic_regression.py     modèle 1 (héritage, encapsulation)
│   ├── neural_network.py          modèle 2 (héritage, encapsulation)
│   └── optimizer.py               descente de gradient
├── pipeline/
│   ├── trainer.py                 boucle d'entraînement
│   └── evaluator.py               accuracy, precision, recall, F1
├── utils/
│   └── metrics.py                 calcul des scores
├── app/
│   └── interface_clinique.py      ClinicalPredictor (interface médecins)
└── main.py                        lance tout le pipeline
```