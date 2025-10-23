import glob, random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC


def train(
    num_categories: int = 5, 
    num_images_per_category: int = 1000, 
    num_pca_components: int = 5, 
    data_location: str = '../dataset'
) -> list:
    X, y = load_data(data_location, num_categories, num_images_per_category)
    X_train, X_test, y_train, y_test = reduce_dimension(X, y, num_pca_components)
    model = train_qsvc(X_train, y_train, num_pca_components)
    y_pred, y_prob = make_prediction(model, X_test)
    print_statistics(y_test, y_pred, y_prob)

    categories = sorted(list(set(y)))

    return categories


def load_data(data_location: str, num_categories: int, num_images_per_category: int) -> tuple:
    X, y = [], []
    files = random.sample(glob.glob(f'{data_location}/*.npy'), k=num_categories)
    for file in files:
        category = file.split('/')[-1].split('\\')[-1].split('.')[0].title()
        images = np.load(file)
        num_images = images.shape[0]
        images_flat = random.sample(images.reshape(num_images, -1).tolist(), k=num_images_per_category)

        X.extend(images_flat)
        y.extend([category] * len(images_flat))

    # print("Data loading complete.")

    return X, y


def reduce_dimension(X: list, y: list, num_pca_components: int) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=num_pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # print("Dimension reduction complete.")

    return X_train_pca, X_test_pca, y_train, y_test


def train_qsvc(X: np.ndarray, y: list, num_pca_components: int) -> QSVC:
    feature_map = ZZFeatureMap(feature_dimension=num_pca_components, reps=2, entanglement='linear')
    kernel = FidelityStatevectorKernel(feature_map=feature_map)
    model = QSVC(quantum_kernel=kernel, probability=True)

    model.fit(X, y)

    # print("QSVC training complete.")

    return model


def make_prediction(model: QSVC, X: np.ndarray) -> tuple:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    # print("Prediction complete.")

    return y_pred, y_prob


def print_statistics(y_test: list, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    y_test = np.array(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    # print(f"AUC: {auc:.3f}")
    # print("Classification Report:\n", report)
    # print("Confusion Matrix:\n", cm)

    return
