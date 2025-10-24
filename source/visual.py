import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

from qiskit import QuantumCircuit
from qiskit_machine_learning.algorithms.classifiers import QSVC


save_location = '../model/'
plt.style.use('dark_background')


def report_statistics(y_test: list, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    y_test = np.array(y_test)

    accuracy = accuracy_score(y_test, y_pred) * 100.0
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr') * 100.0
    report = classification_report(y_test, y_pred)

    with open(save_location + 'statistics.txt', 'w') as file:
        file.write(f"Accuracy: {accuracy:.3f}%\n")
        file.write(f"AUC: {auc:.3f}%\n")
        file.write(f"Classification Report:\n{report}")

    print(f"\nAccuracy: {accuracy:.3f}%")
    print(f"AUC: {auc:.3f}%")
    print(f"Classification Report:\n", report)

    return


def save_feature_map(feature_map: QuantumCircuit) -> None:
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    style = {'name': 'iqp-dark'}
    detailed_feature_map = feature_map.decompose()
    detailed_feature_map.draw(output='mpl', style=style, fold=-1, filename=save_location + 'feature_map.png')

    print("Feature map saved.")

    return


def plot_confusion_matrix(y_test: list, y_pred: np.ndarray, categories: list) -> None:
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('QSVC Confusion Matrix')
    plt.savefig(save_location + 'confusion_matrix.png')
    plt.close()

    print("Confusion matrix saved.")

    return


def plot_qsvc_decision_region(
        model: QSVC, 
        X: list, 
        y: list, 
        categories: list, 
        pca: PCA = None, 
        scaler_std: StandardScaler = None, 
        scaler_quantum: MinMaxScaler = None,
        subset_size=500,
        grid_points_per_axis=100
    ) -> None:
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    X = np.array(X)
    y = np.array(y)

    # PCA transform for visualization.

    if pca is not None and pca.n_components_ >= 2 and scaler_std is not None:
        X_vis = pca.transform(scaler_std.transform(X))[:, :2]
    else:
        X_vis = X[:, :2]

    # Take subset for speed.

    X_vis_sub, _, y_sub, _ = train_test_split(X_vis, y, train_size=subset_size, stratify=y)

    # Mesh grid in PCA 2D space.

    x_min, x_max = X_vis[:, 0].min() - 1.0, X_vis[:, 0].max() + 1.0
    y_min, y_max = X_vis[:, 1].min() - 1.0, X_vis[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points_per_axis),
                         np.linspace(y_min, y_max, grid_points_per_axis))

    # Prepare grid points for QSVC prediction.

    num_grid = xx.ravel().shape[0]
    num_features = model.quantum_kernel.feature_map.num_parameters
    if num_features > 2:
        pad_width = num_features - 2
        grid_points = np.c_[xx.ravel(), yy.ravel(), np.zeros((num_grid, pad_width))]
    else:
        grid_points = np.c_[xx.ravel(), yy.ravel()]

    if scaler_quantum is not None:
        grid_points = scaler_quantum.transform(grid_points)

    # Make predictions on grid points.

    Z = model.predict(grid_points)

    # Encode labels consistently.

    le = LabelEncoder()
    le.fit(categories)
    Z_encoded = le.transform(Z).reshape(xx.shape)
    y_sub_encoded = le.transform(y_sub)

    # Define consistent colormap.

    cmap = ListedColormap(['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD'])

    # Plot decision regions.

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_encoded, alpha=0.3, cmap=cmap)
    for i, category in enumerate(categories):
        idx = y_sub_encoded == i
        plt.scatter(X_vis_sub[idx, 0], X_vis_sub[idx, 1], label=category,
                    color=cmap(i), edgecolor='k', alpha=1.0)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('QSVC Decision Regions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, 'qsvc_decision_regions.png'))
    plt.close()

    print("QSVC decision regions saved.")

    return
