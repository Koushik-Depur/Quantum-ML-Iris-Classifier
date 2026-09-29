import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler


# ==========================================
# 1. Load Iris Dataset
# ==========================================

iris = load_iris()

X = iris.data
y = iris.target

print("Original dataset shape:", X.shape)


# ==========================================
# 2. Select Two Features
# ==========================================

# Feature 0 = Sepal Length
# Feature 2 = Petal Length

X = X[:, [0, 2]]


# ==========================================
# 3. Select Two Classes
# ==========================================

# Class 0 = Setosa
# Class 1 = Versicolor

mask = y < 2

X = X[mask]
y = y[mask]


# ==========================================
# 4. Scale Features for Quantum Circuit
# ==========================================

scaler = MinMaxScaler(feature_range=(0, np.pi))

X = scaler.fit_transform(X)


# ==========================================
# 5. Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Quantum Feature Map
# ==========================================

feature_map = zz_feature_map(
    feature_dimension=2,
    reps=2
)


# ==========================================
# 7. Variational Quantum Circuit
# ==========================================

ansatz = real_amplitudes(
    num_qubits=2,
    reps=2
)


# ==========================================
# 8. Quantum ML Classifier
# ==========================================

sampler = StatevectorSampler()

optimizer = COBYLA(
    maxiter=30
)

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)


# ==========================================
# 9. Train Quantum Model
# ==========================================

print("\nTraining Quantum ML model...")
print("Please wait...\n")

vqc.fit(X_train, y_train)


# ==========================================
# 10. Calculate Accuracy
# ==========================================

accuracy = vqc.score(X_test, y_test)


# ==========================================
# 11. Make Predictions
# ==========================================

predictions = vqc.predict(X_test)


# ==========================================
# 12. Display Results
# ==========================================

print("\n==============================")
print("Quantum ML Results")
print("==============================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")


print("\nPredictions:")
print(predictions)

print("\nActual Labels:")
print(y_test)


# ==========================================
# 13. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 14. Save Confusion Matrix
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Setosa", "Versicolor"]
)

display.plot()

plt.title("Quantum ML - Iris Classification")

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved to:")
print("results/confusion_matrix.png")