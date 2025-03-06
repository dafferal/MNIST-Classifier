from utils import *
import numpy as np
import joblib  # For saving and loading PCA models

class PCAModel:
    """
       The PCA model to reduce the dimensionality of input data
       This standardizes the data, calculates principal components, and projects data onto these components
    """
    def __init__(self, n_components):
        """
            Initialize PCA model with the number of components to retain
            Args:
                n_components (int): Number of principal components to hold
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
            Fits the model onto the training data
            Data is transposed for easier calculations

            Steps:
            1. Standardize the input data
            2. Calculate the covariance matrix
            3. Compute eigenvalues and eigenvectors of the covariance matrix
            4. Sort eigenvectors based on eigenvalues in descending order
            5. Select the top n_components eigenvectors as principal components

            Args:
                X (ndarray): Input training data of shape (n_samples, n_features).
        """

        # Computes the mean for each feature
        self.mean = np.mean(X, axis=0)

        # Computes the standard deviation for each feature
        std = np.std(X, axis=0)

        # Replaces any zero standard deviation with 1 to avoid division by zero
        for i in range(len(std)):
            if std[i] == 0:
                std[i] = 1

        # Standardizes the data
        X = (X - self.mean) / std

        # Calculating the covariance
        covariance = np.cov(X.T)

        # Calculating eigenvectors and eigenvalues
        e_values, e_vectors = np.linalg.eig(covariance)
        e_vectors = e_vectors.T

        # Sorting the calculated eigenvectors
        indices = np.argsort(e_values)[::-1] # Decreasing order
        e_values = e_values[indices]
        e_vectors = e_vectors[indices]

        self.components = e_vectors[:self.n_components]

    def transform(self, X):
        # Computes the mean for each feature
        self.mean = np.mean(X, axis=0)

        # Computes the standard deviation for each feature
        std = np.std(X, axis=0)

        # Replaces any zero standard deviation with 1 to avoid division by zero
        for i in range(len(std)):
            if std[i] == 0:
                std[i] = 1

        # Standardize the data 
        X = (X - self.mean) / std

        return np.dot(X, self.components.T)

def save_PCA(pca, filename='trained_PCA.pkl'):
    # Save the model to a file
    joblib.dump(pca, filename)
    print(f"PCA saved to {filename}")

def load_PCA(filename='trained_PCA.pkl'):
    # Load model from file with error handling if file is missing
    try:
        pca = joblib.load(filename)
        print(f"PCA loaded from {filename}")
        return pca
    except FileNotFoundError:
        return None


def image_to_reduced_feature(images, split='train'):
    """
        Reduce the dimensionality of images using PCA

        Args:
            images (ndarray): Input image data
            split (str): 'train' for training data

        Returns:
            ndarray: The dimensionally reduced features
    """
    globalPCA = load_PCA()

    if globalPCA is None:
        pca = PCAModel(70)
        pca.fit(images)
        reduced_features = pca.transform(images)
        save_PCA(pca)
    else:
        # Adjust inputted data using the obtained values from training
        reduced_features = globalPCA.transform(images)

    return reduced_features

def cosine_similarity(x1, x2):
    """
        Compute cosine similarity between two vectors

        Args:
            x1 (ndarray): First vector
            x2 (ndarray): Second vector

        Returns:
            float: Cosine similarity score
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


class KNNModel:
    """
        A KNN classifier using cosine similarity and weighted similarity
    """
    def __init__(self, train_features, train_labels, k_value):
        """
              Initialising the KNN model

              Args:
                  train_features (ndarray): Training the feature vectors
                  train_labels (ndarray): The labels for the training data
                  k_value (int): Number of neighbors
        """
        self.X = train_features
        self.y = train_labels
        self.k = k_value

        # Error checking
        if self.k <= 0:
            raise ValueError("Invalid k value")

    def fit(self, X, y):
        """
            Store training data and labels

            Args:
                X (ndarray): Training feature vectors
                y (ndarray): Training labels
        """
        self.X = np.copy(X)
        self.y = np.copy(y)

    def predict(self, X):
        """
            Predict the labels for test data using cosine similarity and weighted similarity

            Args:
                X (ndarray): Test feature vectors, from PCA

            Returns:
                list: Predicted labels for the test data
        """
        predictions = []
        for x in X:
            # Compute cosine similarity between x and all training values
            similarities = []
            for x_train in self.X:
                similarity = cosine_similarity(x, x_train)
                similarities.append(similarity)

            # Get the indices of the k nearest neighbor
            k_indices = np.argsort(similarities)[-self.k:]  # The K largest similarities

            # Get the labels of the k neighbors
            k_nearest_labels = []
            for i in k_indices:
                k_nearest_labels.append(self.y[i])
            # Get the similarities of the k neighbors
            k_nearest_similarities = []
            for i in k_indices:
                k_nearest_similarities.append(similarities[i])

            # Find the most weighted label
            highest_weights = {}
            for i in range(len(k_nearest_labels)):
                label = k_nearest_labels[i]
                similarity = k_nearest_similarities[i]

                if label not in highest_weights:
                    highest_weights[label] = 0  # 0 if not in the list already
                highest_weights[label] += similarity  # Add the weight

            # Find the label with the highest weight
            most_common_label = None
            highest_weight = -1  # Starting value of -1

            # Find the label with the maximum weight
            for label, weight in highest_weights.items():
                if weight > highest_weight:
                    most_common_label = label
                    highest_weight = weight

            predictions.append(most_common_label)
        return predictions

def training_model(train_features, train_labels):
    """
        Initialize and return the KNN model

        Args:
            train_features (ndarray): Training feature vectors
            train_labels (ndarray): Training labels

        Returns:
            KNNModel: A trained KNN model, given a k value
    """
    return KNNModel(train_features, train_labels, 6)

