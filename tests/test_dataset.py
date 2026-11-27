from core.dataset import Dataset


def test_dataset_charge_300_patients():
    dataset = Dataset("data/patient_data.csv")
    assert len(dataset) == 300


def test_chaque_patient_a_4_caracteristiques():
    dataset = Dataset("data/patient_data.csv")
    x, y = dataset[0]
    assert len(x) == 4
    assert y in (0, 1)


def test_split_80_20():
    dataset = Dataset("data/patient_data.csv")
    X_train, X_test, y_train, y_test = dataset.split(test_ratio=0.2)
    assert len(X_train) == 240
    assert len(X_test) == 60
    assert len(X_train) == len(y_train)
