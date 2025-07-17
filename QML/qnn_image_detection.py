from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import Sampler
import time
import numpy as np

class QNNImageClassifier:
    def __init__(self, n_qubits=None, n_features=16, optimizer='SPSA', max_iter=100):
        if n_qubits is None:
            self.n_qubits = n_features
        else:
            if n_qubits < n_features:
                print(f"Warning: n_qubits ({n_qubits}) < n_features ({n_features}). Using n_features for qubit count.")
            self.n_qubits = max(n_qubits, n_features)

        self.n_features = n_features
        self.max_iter = max_iter

        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.decomposition import PCA
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_features)
        self.feature_scaler = MinMaxScaler(feature_range=(0, np.pi))

        if optimizer == 'SPSA':
            self.optimizer = SPSA(maxiter=max_iter, learning_rate=0.01, perturbation=0.01)
        else:
            self.optimizer = COBYLA(maxiter=max_iter)

        self._setup_quantum_circuit()

    def _setup_quantum_circuit(self):
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_features,
            reps=1,
            entanglement='linear'
        )

        self.ansatz = RealAmplitudes(
            num_qubits=self.n_features,
            reps=2,
            entanglement='full'
        )

        self.qc = QuantumCircuit(self.n_features)
        self.qc.compose(self.feature_map, inplace=True)
        self.qc.compose(self.ansatz, inplace=True)

        qc_with_measurements = self.qc.copy()
        qc_with_measurements.measure_all()

        self.sampler = Sampler()

        self.qnn = SamplerQNN(
            circuit=qc_with_measurements,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            sampler=self.sampler
        )

        # interpretation is removed as it's deprecated
        self.classifier = NeuralNetworkClassifier(
            neural_network=self.qnn,
            optimizer=self.optimizer
        )

    def preprocess_data(self, X, y=None, fit_transform=True):
        if fit_transform:
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)
            X_processed = self.feature_scaler.fit_transform(X_pca)
        else:
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled)
            X_processed = self.feature_scaler.transform(X_pca)

        if y is not None:
            y_processed = (y > y.mean()).astype(int)
            return X_processed, y_processed
        return X_processed

    def train(self, X_train, y_train, verbose=True):
        start_time = time.time()
        X_processed, y_processed = self.preprocess_data(X_train, y_train, fit_transform=True)

        if verbose:
            print(f"Training QNN with {len(X_processed)} samples...")
            print(f"Feature dimension: {self.n_features}, Qubits: {self.n_qubits}")

        self.classifier.fit(X_processed, y_processed)
        training_time = time.time() - start_time

        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")

        return training_time

    def predict(self, X_test):
        X_processed = self.preprocess_data(X_test, fit_transform=False)
        return self.classifier.predict(X_processed)

    def evaluate(self, X_test, y_test, verbose=True):
        start_time = time.time()
        _, y_processed = self.preprocess_data(X_test, y_test, fit_transform=False)
        predictions = self.predict(X_test)

        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_processed, predictions)
        prediction_time = time.time() - start_time

        if verbose:
            print(f"\n=== QNN Performance Report ===")
            print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Prediction Time: {prediction_time:.2f} seconds")
            print(f"Samples/sec: {len(X_test)/prediction_time:.2f}")
            print("\n" + classification_report(y_processed, predictions))

        return accuracy, prediction_time

def load_and_prepare_data():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X, y = digits.data, digits.target
    y_binary = (y % 2 == 0).astype(int)

    return train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

def demonstrate_qnn_performance():
    print("Quantum Neural Network Image Detection Demo")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}")

    qnn_classifier = QNNImageClassifier(
        n_features=8,
        optimizer='SPSA',
        max_iter=50
    )

    print(f"Initialized QNN with {qnn_classifier.n_qubits} qubits and {qnn_classifier.n_features} features.")
    training_time = qnn_classifier.train(X_train, y_train)
    accuracy, prediction_time = qnn_classifier.evaluate(X_test, y_test)

    print(f"\nSummary:")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Prediction Speed: {len(X_test)/prediction_time:.2f} samples/sec")
    print(f"Final Accuracy: {accuracy*100:.2f}%")

    return qnn_classifier, accuracy

def optimize_for_higher_accuracy():
    print("\n Optimizing for Higher Accuracy...")
    print("=" * 40)

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    qnn_optimized = QNNImageClassifier(
        n_features=12,
        optimizer='COBYLA',
        max_iter=100
    )

    print(f"Initialized QNN with {qnn_optimized.n_qubits} qubits and {qnn_optimized.n_features} features.")
    training_time = qnn_optimized.train(X_train, y_train)
    accuracy, prediction_time = qnn_optimized.evaluate(X_test, y_test)

    return qnn_optimized, accuracy

if __name__ == "__main__":
    classifier, acc1 = demonstrate_qnn_performance()

    if acc1 < 0.75:
        print("\n Accuracy below target, trying optimized configuration...")
        classifier_opt, acc2 = optimize_for_higher_accuracy()

        if acc2 >= 0.75:
            print(" Optimized version achieved target accuracy!")
        else:
            print(" Consider increasing n_features or max_iter for better accuracy")

    print("\n QNN Image Detection Demo Complete!")
