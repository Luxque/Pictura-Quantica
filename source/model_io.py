import os, joblib

from qiskit_machine_learning.algorithms.classifiers import QSVC


def save_model(model: QSVC, categories: list) -> None:
    save_dir = '../model/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    joblib.dump(model, save_dir + 'qsvm_model.pkl')
    joblib.dump(categories, save_dir + 'categories.pkl')

    print("Saving complete.")

    return


def load_model() -> tuple:
    save_dir = '../model/'
    if not os.path.exists(save_dir):
        raise RuntimeError("Model directory does not exist.")

    model = joblib.load(save_dir + 'qsvm_model.pkl')
    categories = joblib.load(save_dir + 'categories.pkl')

    print("Loading complete.")

    return model, categories
