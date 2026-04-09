--> Hybrid Quantum Machine Learning for Drug Discovery: Molecular Activity Prediction


--> Overview

- This project explores the use of hybrid quantum-classical machine learning for biomedical prediction tasks. A hybrid model combining classical neural networks with a quantum layer was developed to classify tumors as malignant or benign using a real-world dataset.
- The goal is to evaluate whether quantum machine learning (QML) can provide meaningful improvements over traditional models in healthcare-related prediction tasks.

--> Key Features

- Hybrid Quantum + PyTorch model using PennyLane
- Comparison with classical models (Logistic Regression, Random Forest)
- Dimensionality reduction using PCA for quantum compatibility
- Performance evaluation using Accuracy, Precision, Recall, and F1 Score
- Visualization of model performance and training behavior

--> Tech Stack
      - Python
      - Pandas
      - NumPy
      - Scikit-learn
      - PyTorch
      - PennyLane
      - Matplotlib



--> Methodology

1. Data Preprocessing
Loaded breast cancer dataset
Performed train-test split
Applied feature scaling using StandardScaler
Reduced dimensions to 4 features using PCA (to fit quantum circuit constraints)

2. Classical Models
Logistic Regression
Random Forest

3. Hybrid Quantum Model
Classical linear layer
Quantum circuit layer (AngleEmbedding + Entangling layers)
Final classification layer with sigmoid activation

--> Results

Model	              Accuracy	Precision  Recall	     F1 Score
Logistic Regression	 0.9649	   0.9722	   0.9722	     0.9722
Random Forest	       0.9210	   0.9436	   0.9306	     0.9371
Hybrid Quantum Model 0.9474	   0.9342	   0.9861	     0.9595


--> Key Insights

- Logistic Regression achieved the highest overall accuracy
- The hybrid quantum model achieved the highest recall (98.6%)
- High recall is critical in healthcare applications, as it minimizes missed positive cases (false negatives)
- The results demonstrate that hybrid quantum models can perform competitively with classical methods


--> Project Structure

Hybrid-Quantum-Machine-Learning-for-Drug-Discovery/
│── results/
│   │── model_metrics_comparison.csv
│   │── classification_reports.txt
│   │── confusion_matrices.txt
│   │── model_accuracy_comparison.png
│   │── quantum_training_loss.png
│── main.py
│── README.md
│── requirements.txt


--> How to Run

1. Install dependencies
pip install pandas numpy matplotlib scikit-learn torch pennylane

2. Run the project
python main.py

3. View results


--> Future Improvements

- Use real molecular datasets (e.g., MoleculeNet, BACE)
- Experiment with deeper quantum circuits
- Compare different quantum embeddings
- Scale hybrid models with larger datasets


--> Conclusion

This project demonstrates how quantum machine learning can be integrated into real-world pipelines. 
While classical models still perform strongly, the hybrid quantum model shows promising results, particularly in high-recall scenarios critical for healthcare and drug discovery.

--> Author
----- Devarshi Trivedi || UT Dallas
