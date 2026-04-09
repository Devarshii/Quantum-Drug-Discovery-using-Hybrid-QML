import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

# ============================================
# Create folders
# ============================================
os.makedirs("results", exist_ok=True)

# ============================================
# Load dataset
# ============================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Original dataset shape:", X.shape)
print("Target shape:", y.shape)

# ============================================
# Train-test split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# Feature scaling
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# PCA for dimensionality reduction
# Quantum circuits work better with fewer features
# ============================================
n_qubits = 4
pca = PCA(n_components=n_qubits)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Reduced train shape:", X_train_pca.shape)
print("Reduced test shape:", X_test_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", np.sum(pca.explained_variance_ratio_))

# Save explained variance
with open("results/pca_info.txt", "w") as f:
    f.write("Explained variance ratio:\n")
    f.write(str(pca.explained_variance_ratio_))
    f.write("\n\nTotal explained variance:\n")
    f.write(str(np.sum(pca.explained_variance_ratio_)))

# ============================================
# Classical Models
# ============================================
print("\nRunning Logistic Regression...")
log_model = LogisticRegression(random_state=42, max_iter=2000)
log_model.fit(X_train_pca, y_train)
y_pred_log = log_model.predict(X_test_pca)

log_acc = accuracy_score(y_test, y_pred_log)
log_precision = precision_score(y_test, y_pred_log)
log_recall = recall_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log)

print("Logistic Regression Accuracy:", log_acc)
print(classification_report(y_test, y_pred_log))

print("\nRunning Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_train)
y_pred_rf = rf_model.predict(X_test_pca)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print("Random Forest Accuracy:", rf_acc)
print(classification_report(y_test, y_pred_rf))

# ============================================
# Prepare data for PyTorch
# ============================================
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# ============================================
# Quantum Layer Setup
# ============================================
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (2, n_qubits, 3)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# ============================================
# Hybrid Quantum-Classical Model
# ============================================
class HybridQuantumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_qubits, n_qubits)
        self.quantum = quantum_layer
        self.fc2 = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.quantum(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = HybridQuantumModel()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ============================================
# Train Hybrid Model
# ============================================
epochs = 30
loss_history = []

print("\nTraining Hybrid Quantum Model...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ============================================
# Evaluate Hybrid Model
# ============================================
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    y_pred_q = (y_pred_probs >= 0.5).float().numpy().flatten()

q_acc = accuracy_score(y_test, y_pred_q)
q_precision = precision_score(y_test, y_pred_q)
q_recall = recall_score(y_test, y_pred_q)
q_f1 = f1_score(y_test, y_pred_q)

print("\nHybrid Quantum Model Accuracy:", q_acc)
print(classification_report(y_test, y_pred_q))

# ============================================
# Save metrics comparison
# ============================================
metrics_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Hybrid Quantum Model"],
    "Accuracy": [log_acc, rf_acc, q_acc],
    "Precision": [log_precision, rf_precision, q_precision],
    "Recall": [log_recall, rf_recall, q_recall],
    "F1 Score": [log_f1, rf_f1, q_f1]
})

metrics_df.to_csv("results/model_metrics_comparison.csv", index=False)
print("\nSaved: results/model_metrics_comparison.csv")

# ============================================
# Save classification reports
# ============================================
with open("results/classification_reports.txt", "w") as f:
    f.write("=== Logistic Regression ===\n")
    f.write(classification_report(y_test, y_pred_log))
    f.write("\n\n=== Random Forest ===\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write("\n\n=== Hybrid Quantum Model ===\n")
    f.write(classification_report(y_test, y_pred_q))

print("Saved: results/classification_reports.txt")

# ============================================
# Save confusion matrices
# ============================================
cm_log = confusion_matrix(y_test, y_pred_log)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_q = confusion_matrix(y_test, y_pred_q)

with open("results/confusion_matrices.txt", "w") as f:
    f.write("Logistic Regression Confusion Matrix:\n")
    f.write(str(cm_log))
    f.write("\n\nRandom Forest Confusion Matrix:\n")
    f.write(str(cm_rf))
    f.write("\n\nHybrid Quantum Model Confusion Matrix:\n")
    f.write(str(cm_q))

print("Saved: results/confusion_matrices.txt")

# ============================================
# Plot 1: Model Accuracy Comparison
# ============================================
plt.figure(figsize=(8, 5))
plt.bar(metrics_df["Model"], metrics_df["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/model_accuracy_comparison.png")
plt.show()

print("Saved: results/model_accuracy_comparison.png")

# ============================================
# Plot 2: Training Loss Curve
# ============================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Hybrid Quantum Model Training Loss")
plt.tight_layout()
plt.savefig("results/quantum_training_loss.png")
plt.show()

print("Saved: results/quantum_training_loss.png")

# ============================================
# Final print summary
# ============================================
print("\n================ FINAL RESULTS ================")
print(metrics_df)
print("==============================================")