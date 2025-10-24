import os, joblib

from qiskit_machine_learning.algorithms.classifiers import QSVC

save_location = '../model/'


def save_model(model: QSVC, categories: list) -> None:
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    joblib.dump(model, save_location + 'model.pkl')
    joblib.dump(categories, save_location + 'categories.pkl')

    print("Saving complete.")

    return


def load_model() -> tuple:
    if not os.path.exists(save_location):
        raise RuntimeError("Model directory does not exist.")

    model = joblib.load(save_location + 'model.pkl')
    categories = joblib.load(save_location + 'categories.pkl')

    print("Loading complete.")

    return model, categories
