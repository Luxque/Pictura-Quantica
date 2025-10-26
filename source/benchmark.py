import time, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from train import load_data, reduce_dimension, train_qsvc, make_prediction


save_location = '../figures'
data_location = '../dataset'
num_categories = 5
num_images_per_category = 500
num_pca_components = 6
reps = 1
entanglement = 'linear'
repetition_per_test = 5

plt.style.use('dark_background')


def measure_num_categories() -> tuple[list, list, list]:
    tests, accuracies, elapsed_times = [], [], []
    for num_categories in tqdm(range(2, 10+1)):
        accuracies_container, elapsed_times_container = [], []
        for _ in range(repetition_per_test):
            start_time = time.time()
            X, y = load_data(data_location, num_categories, num_images_per_category)
            X_train, X_test, y_train, y_test, _, _, _ = reduce_dimension(X, y, num_pca_components)
            model, _ = train_qsvc(X_train, y_train, num_pca_components, reps, entanglement)
            y_pred, _ = make_prediction(model, X_test)
            accuracy = compute_accuracy(y_test, y_pred)
            end_time = time.time()

            accuracies_container.append(accuracy)
            elapsed_times_container.append(end_time - start_time)

        tests.append(num_categories)
        accuracies.append(np.average(accuracies_container))
        elapsed_times.append(np.average(elapsed_times_container))

    print("Measuring accuracies and elapsed times over `num_categories` complete.")
    
    return tests, accuracies, elapsed_times


def measure_num_images_per_category() -> tuple[list, list, list]:
    tests, accuracies, elapsed_times = [], [], []
    for num_images_per_category in tqdm(range(100, 1000+1, 50)):
        accuracies_container, elapsed_times_container = [], []
        for _ in range(repetition_per_test):
            start_time = time.time()
            X, y = load_data(data_location, num_categories, num_images_per_category)
            X_train, X_test, y_train, y_test, _, _, _ = reduce_dimension(X, y, num_pca_components)
            model, _ = train_qsvc(X_train, y_train, num_pca_components, reps, entanglement)
            y_pred, _ = make_prediction(model, X_test)
            accuracy = compute_accuracy(y_test, y_pred)
            end_time = time.time()

            accuracies_container.append(accuracy)
            elapsed_times_container.append(end_time - start_time)

        tests.append(num_images_per_category)
        accuracies.append(np.average(accuracies_container))
        elapsed_times.append(np.average(elapsed_times_container))

    print("Measuring accuracies and elapsed times over `num_images_per_category` complete.")
    
    return tests, accuracies, elapsed_times


def measure_num_pca_components() -> tuple[list, list, list]:
    tests, accuracies, elapsed_times = [], [], []
    for num_pca_components in tqdm(range(2, 10+1)):
        accuracies_container, elapsed_times_container = [], []
        for _ in range(repetition_per_test):
            start_time = time.time()
            X, y = load_data(data_location, num_categories, num_images_per_category)
            X_train, X_test, y_train, y_test, _, _, _ = reduce_dimension(X, y, num_pca_components)
            model, _ = train_qsvc(X_train, y_train, num_pca_components, reps, entanglement)
            y_pred, _ = make_prediction(model, X_test)
            accuracy = compute_accuracy(y_test, y_pred)
            end_time = time.time()

            accuracies_container.append(accuracy)
            elapsed_times_container.append(end_time - start_time)

        tests.append(num_pca_components)
        accuracies.append(np.average(accuracies_container))
        elapsed_times.append(np.average(elapsed_times_container))

    print("Measuring accuracies and elapsed times over `num_pca_components` complete.")
    
    return tests, accuracies, elapsed_times


def measure_reps() -> tuple[list, list, list]:
    tests, accuracies, elapsed_times = [], [], []
    for reps in tqdm(range(2, 10+1)):
        accuracies_container, elapsed_times_container = [], []
        for _ in range(repetition_per_test):
            start_time = time.time()
            X, y = load_data(data_location, num_categories, num_images_per_category)
            X_train, X_test, y_train, y_test, _, _, _ = reduce_dimension(X, y, num_pca_components)
            model, _ = train_qsvc(X_train, y_train, num_pca_components, reps, entanglement)
            y_pred, _ = make_prediction(model, X_test)
            accuracy = compute_accuracy(y_test, y_pred)
            end_time = time.time()

            accuracies_container.append(accuracy)
            elapsed_times_container.append(end_time - start_time)

        tests.append(reps)
        accuracies.append(np.average(accuracies_container))
        elapsed_times.append(np.average(elapsed_times_container))

    print("Measuring accuracies and elapsed times over `reps` complete.")
    
    return tests, accuracies, elapsed_times


def measure_entanglement() -> tuple[list, list, list]:
    tests, accuracies, elapsed_times = [], [], []
    for id, entanglement in enumerate(tqdm(['full', 'linear', 'reverse_linear', 'circular', 'sca'])):
        accuracies_container, elapsed_times_container = [], []
        for _ in range(repetition_per_test):
            start_time = time.time()
            X, y = load_data(data_location, num_categories, num_images_per_category)
            X_train, X_test, y_train, y_test, _, _, _ = reduce_dimension(X, y, num_pca_components)
            model, _ = train_qsvc(X_train, y_train, num_pca_components, reps, entanglement)
            y_pred, _ = make_prediction(model, X_test)
            accuracy = compute_accuracy(y_test, y_pred)
            end_time = time.time()

            accuracies_container.append(accuracy)
            elapsed_times_container.append(end_time - start_time)

        tests.append(id)
        accuracies.append(np.average(accuracies_container))
        elapsed_times.append(np.average(elapsed_times_container))

    print("Measuring accuracies and elapsed times over `entanglement` complete.")
    
    return tests, accuracies, elapsed_times


def compute_accuracy(y_test: list, y_pred: np.ndarray) -> float:
    return accuracy_score(y_test, y_pred) * 100.0


def plot_results(tests: list, accuracies: list, elapsed_times: list, x_label: str, filename: str) -> None:
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    # Define colors.

    color_accuracies = 'tab:cyan'
    color_elapsed_times = 'tab:orange'

    # Configure axis 1.

    _, axis1 = plt.subplots(figsize=(8, 6))
    axis1.set_xlabel(x_label)
    axis1.set_ylabel('Accuracy (%)', color=color_accuracies)
    axis1.plot(tests, accuracies, color=color_accuracies, label='Accuracy')
    axis1.tick_params(axis='y', labelcolor=color_accuracies)
    axis1.grid(True, alpha=0.3)

    # Configure axis 2.

    axis2 = axis1.twinx()
    axis2.set_ylabel('Elapsed Time (sec)', color=color_elapsed_times)
    axis2.plot(tests, elapsed_times, color=color_elapsed_times, label='Time')

    # Handle labels.

    lines1, labels1 = axis1.get_legend_handles_labels()
    lines2, labels2 = axis2.get_legend_handles_labels()
    axis1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Plot the resuls.

    plt.title(f"QSVC Benchmark: {x_label} vs. Accuracy and Time")
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, filename))
    plt.close()

    return


"""
Ceteris paribus testing approach.
"""
def main() -> None:
    test, accuracies, elapsed_times = measure_num_categories()
    plot_results(test, accuracies, elapsed_times, 'Number of Categories', 'num_of_categories.png')
    print("Benchmarking `num_of_categories` complete.")

    test, accuracies, elapsed_times = measure_num_images_per_category()
    plot_results(test, accuracies, elapsed_times, 'Number of Images per Category', 'num_images_per_category.png')
    print("Benchmarking `num_images_per_category` complete.")

    test, accuracies, elapsed_times = measure_num_pca_components()
    plot_results(test, accuracies, elapsed_times, 'Number of PCA Components', 'num_pca_components.png')
    print("Benchmarking `num_pca_components` complete.")

    test, accuracies, elapsed_times = measure_reps()
    plot_results(test, accuracies, elapsed_times, 'Number of Repetition', 'reps.png')
    print("Benchmarking `reps` complete.")

    test, accuracies, elapsed_times = measure_entanglement()
    plot_results(test, accuracies, elapsed_times, 'Entanglement Structure ID', 'entanglement.png')
    print("Benchmarkign `entanglement` complete.")

    return


if __name__ == '__main__':
    main()
