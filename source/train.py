import glob, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import cv2


def train(file_path: str) -> None:
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = reduce_dimension(data)

    return


def load_data(file_path: str) -> dict:
    data = {}

    files = glob.glob(file_path + '/*.npy')
    for file in files:
        category = file.split('/')[-1].split('\\')[-1].split('.')[0].title()
        images = np.load(file)
        data[category] = images
    
    return data


def reduce_dimension(data: dict) -> tuple:
    X, y = [], []
    for category, images in data.items():
        num_images = images.shape[0]
        images_flat = images.reshape(num_images, 28, 28, -1)
        print(images_flat.shape)
        exit(0)
        
        # image = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
        # cv2.imshow("Sample Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        X.append(images_flat)
        y.extend([category] * num_images)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=6)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, y_train, y_test