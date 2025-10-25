import os, joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from qiskit_machine_learning.algorithms.classifiers import QSVC

save_location = '../model/'


def save_model(model: QSVC, categories: list, pca: PCA, scaler_std: StandardScaler, scaler_quantum: MinMaxScaler) -> None:
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    joblib.dump(model, save_location + 'model.pkl')
    joblib.dump({
        'categories': categories,
        'pca': pca,
        'scaler_std': scaler_std,
        'scaler_quantum': scaler_quantum
    }, save_location + 'metadata.pkl')

    print("Saving complete.")

    return


def load_model() -> tuple[QSVC, list, PCA, StandardScaler, MinMaxScaler]:
    if not os.path.exists(save_location):
        raise RuntimeError("Model directory does not exist.")

    model = joblib.load(save_location + 'model.pkl')
    metadata = joblib.load(save_location + 'metadata.pkl')

    print("Loading complete.")

    return model, metadata['categories'], metadata['pca'], metadata['scaler_std'], metadata['scaler_quantum']
