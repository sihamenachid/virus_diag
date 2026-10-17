class ClinicalPredictor:
    """
    Question 1 : interface pour les médecins.
    - reçoit un modèle déjà entraîné
    - diagnose(patient_data) retourne "Infecté" si prédiction >= 0.5, "Sain" sinon
    """

    def __init__(self, model, normalizer=None):
        self.model = model              # modèle déjà entraîné
        self.normalizer = normalizer

    def diagnose(self, patient_data):
        x = patient_data
        if self.normalizer is not None:
            x = self.normalizer.transform_one(x)
        prediction = self.model.predict(x)   # polymorphisme : marche avec tout modèle ayant predict()
        if prediction >= 0.5:
            return "Infecté"
        return "Sain"
