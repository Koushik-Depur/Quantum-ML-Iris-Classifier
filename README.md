# ⚛️ Quantum ML Iris Classifier

A beginner-friendly **Quantum Machine Learning** project that uses Qiskit's **Variational Quantum Classifier (VQC)** to classify Iris flowers.

## 📌 Project Overview

This project demonstrates how classical machine-learning data can be encoded into a quantum circuit and used by a variational quantum classifier.

The project uses two Iris flower features:

* Sepal Length
* Petal Length

The classifier distinguishes between:

* Setosa
* Versicolor

The quantum model runs locally using a quantum simulator.

## 🧠 Technologies

* Python
* Qiskit
* Qiskit Machine Learning
* NumPy
* scikit-learn
* Matplotlib

## 🔬 Quantum ML Workflow

```text
Iris Dataset
     ↓
Feature Selection
     ↓
Data Scaling
     ↓
Quantum Feature Map
     ↓
Variational Quantum Circuit
     ↓
Quantum Classifier
     ↓
Prediction
     ↓
Evaluation
```

## ⚛️ Quantum Circuit

The project uses:

### Quantum Feature Map

The input features are encoded into a quantum circuit using a ZZ feature map.

### Variational Ansatz

A trainable Real Amplitudes circuit is used as the variational circuit.

### VQC

The Quantum Machine Learning classifier combines the feature map and variational circuit.

## 📊 Dataset

The Iris dataset contains measurements of Iris flowers.

For this project, only two classes and two features are used to keep the quantum circuit small and easy to understand.

## 📈 Evaluation

The model is evaluated using:

* Test accuracy
* Predictions
* Actual labels
* Confusion matrix

The confusion matrix is automatically saved to:

```text
results/confusion_matrix.png
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Quantum-ML-Iris-Classifier.git
cd Quantum-ML-Iris-Classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the project

```bash
python src/main.py
```

## 📁 Project Structure

```text
Quantum-ML-Iris-Classifier/
│
├── src/
│   └── main.py
│
├── results/
│   └── confusion_matrix.png
│
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

## 🎯 Learning Objectives

This project was created to understand the fundamentals of Quantum Machine Learning, including:

* Qubits
* Quantum feature encoding
* Quantum circuits
* Variational circuits
* Quantum classification
* Classical data preparation
* Model evaluation

## ⚠️ Note

This project is an educational demonstration of Quantum Machine Learning using a simulator. It does not claim that the quantum model outperforms classical machine-learning algorithms.

## 👨‍💻 Author

**Koushik Depur**

Aspiring AI / ML Engineer | Data Science Enthusiast
