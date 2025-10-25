import glob, os, random
import numpy as np

from cv2 import resize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC

from visual import report_statistics, save_feature_map, plot_confusion_matrix, plot_qsvc_decision_region


def load_data(data_location: str, num_categories: int, num_images_per_category: int) -> tuple[list, list]:
    X, y = [], []
    files = random.sample(glob.glob(os.path.join(data_location, '*.npy')), k=num_categories)
    for file in files:
        category = file.split('/')[-1].split('\\')[-1].split('.')[0].title()
        images = np.load(file)
        num_images = images.shape[0]
        images_flat = random.sample(images.reshape(num_images, -1).tolist(), k=num_images_per_category)

        X.extend(images_flat)
        y.extend([category] * len(images_flat))

    print("Data loading complete.")

    return X, y


def reduce_dimension(X: list, y: list, num_pca_components: int) -> tuple[np.ndarray, np.ndarray, list, list, PCA, StandardScaler, MinMaxScaler]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler_std = StandardScaler()
    X_train_scaled = scaler_std.fit_transform(X_train)
    X_test_scaled = scaler_std.transform(X_test)

    pca = PCA(n_components=num_pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    scaler_quantum = MinMaxScaler(feature_range=(0.0, np.pi))
    X_train_quantum = scaler_quantum.fit_transform(X_train_pca)
    X_test_quantum = scaler_quantum.transform(X_test_pca)

    print("Dimension reduction complete.")

    return X_train_quantum, X_test_quantum, y_train, y_test, pca, scaler_std, scaler_quantum


def train_qsvc(X: np.ndarray, y: list, num_pca_components: int, reps: int, entanglement: str) -> tuple[QSVC, QuantumCircuit]:
    feature_map = zz_feature_map(feature_dimension=num_pca_components, reps=reps, entanglement=entanglement)
    kernel = FidelityStatevectorKernel(feature_map=feature_map)
    model = QSVC(quantum_kernel=kernel, probability=True)

    model.fit(X, y)

    print("QSVC training complete.")

    return model, feature_map


def make_prediction(model: QSVC, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    print("Prediction complete.")

    return y_pred, y_prob


def classify_image(image: np.ndarray, model: QSVC, pca: PCA, scaler_std: StandardScaler, scaler_quantum: MinMaxScaler):
    image_resized = resize(image, (28, 28)).flatten()
    image_scaled = scaler_std.transform([image_resized])
    image_pca = pca.transform(image_scaled)
    image_quantum = scaler_quantum.transform(image_pca)     
    classification = model.predict(image_quantum)[0]

    print(f"The image is classified as {classification}.")

    return classification


def train(
    data_location: str = '../dataset', 
    num_categories: int = 5, 
    num_images_per_category: int = 500, 
    num_pca_components: int = 6, 
    reps: int = 1, 
    entanglement: str = 'linear', 
    plots: bool = True
) -> tuple[QSVC, list]:
    X, y = load_data(data_location, num_categories, num_images_per_category)
    X_train, X_test, y_train, y_test, pca, scaler_std, scaler_quantum = reduce_dimension(X, y, num_pca_components)
    model, feature_map = train_qsvc(X_train, y_train, num_pca_components, reps, entanglement)
    y_pred, y_prob = make_prediction(model, X_test)

    categories = sorted(list(set(y)))

    report_statistics(y_test, y_pred, y_prob)
    if plots:
        save_feature_map(feature_map)
        plot_confusion_matrix(y_test, y_pred, categories)
        plot_qsvc_decision_region(model, X, y, categories, pca, scaler_std, scaler_quantum)

    return model, categories, pca, scaler_std, scaler_quantum
