import glob, os, random
import numpy as np

from cv2 import resize
from scipy.ndimage import center_of_mass, shift, rotate, binary_dilation, binary_erosion

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC

from visual import report_statistics, save_feature_map, plot_confusion_matrix, plot_qsvc_decision_region


def random_shift(images, max_shift=2):
    shifted_images = []
    for img_flat in images:
        img = img_flat.reshape(28, 28)
        shift_x = np.random.randint(-max_shift, max_shift+1)
        shift_y = np.random.randint(-max_shift, max_shift+1)
        img_shifted = shift(img, shift=(shift_y, shift_x), mode='nearest')
        shifted_images.append(img_shifted.flatten())
    
    return np.array(shifted_images)


def random_rotate(images, max_angle=10):
    rotated_images = []
    for img_flat in images:
        img = img_flat.reshape(28, 28)
        angle = np.random.uniform(-max_angle, max_angle)
        img_rotated = rotate(img, angle=angle, reshape=False, mode='nearest')
        rotated_images.append(img_rotated.flatten())

    return np.array(rotated_images)


def random_thickness(images, prob_dilate=0.5):
    thick_images = []
    for img_flat in images:
        img = img_flat.reshape(28, 28)
        binary_img = img > 127  # threshold to binary
        if np.random.rand() < prob_dilate:
            img_aug = binary_dilation(binary_img).astype(float) * 255
        else:
            img_aug = binary_erosion(binary_img).astype(float) * 255
        thick_images.append(img_aug.flatten())

    return np.array(thick_images)


def add_noise(images: np.ndarray, std_dev: float = 0.5) -> np.ndarray:
    noisy = images + np.random.normal(0, std_dev, images.shape)
    noisy = np.clip(noisy, 0, 255)

    return noisy


def load_data(data_location: str, num_categories: int, num_images_per_category: int) -> tuple[list, list]:
    X, y = [], []
    files = random.sample(glob.glob(os.path.join(data_location, '*.npy')), k=num_categories)
    for file in files:     
        images = np.load(file)
        num_images = images.shape[0]
        all_images = images.reshape(num_images, -1)
        images_flat = np.array(random.sample(list(all_images), k=num_images_per_category))

        images_noisy = random_shift(images_flat)
        images_noisy = random_rotate(images_noisy)
        images_noisy = random_thickness(images_noisy)
        images_noisy = add_noise(images_noisy)

        category = file.split('/')[-1].split('\\')[-1].split('.')[0].title()

        X.extend(images_noisy)
        y.extend([category] * len(images_noisy))

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


def train(
    data_location: str = '../dataset', 
    num_categories: int = 5, 
    num_images_per_category: int = 1000, 
    num_pca_components: int = 6, 
    reps: int = 1, 
    entanglement: str = 'linear', 
    plots: bool = False
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

    print("Training complete.")

    return model, categories, pca, scaler_std, scaler_quantum


def center_image(image: np.ndarray) -> np.ndarray:
    # Compute center of mass (weighted by intensity)

    cy, cx = center_of_mass(image)
    
    # Check if it is an empty canvas.

    if np.isnan(cx) or np.isnan(cy):
        return image
    
    # Compute the shift needed to move CoM to the image center.

    height, width = image.shape
    shift_x = width / 2 - cx
    shift_y = height / 2 - cy
    
    # Shift the image using nearest-neighbor to avoid interpolation blur.

    centered = shift(image, shift=(shift_y, shift_x), mode='constant', cval=0)
    
    return centered


def classify_image(
        image: np.ndarray, 
        model: QSVC, 
        pca: PCA, 
        scaler_std: StandardScaler, 
        scaler_quantum: MinMaxScaler
    ) -> str:
    image_centered = center_image(image)
    image_resized = resize(image_centered, (28, 28)).flatten()
    if image_resized.max() <= 1.0:
        image_resized *= 255.0

    image_scaled = scaler_std.transform([image_resized])
    image_pca = pca.transform(image_scaled)
    image_quantum = scaler_quantum.transform(image_pca)
    classification = model.predict(image_quantum)[0]

    print(f"The image is classified as {classification}.")

    return classification
