Detecting Parkinson's Disease using Machine Learning
This repository contains a Jupyter Notebook implementation for detecting Parkinson's Disease using a Random Forest Classifier. The project leverages biomedical voice measurements to classify individuals as affected or not affected by Parkinson's Disease.

Table of Contents
Overview
Dataset
Features
Model and Techniques
Results
Dependencies
Usage
Contributing
License
Overview
Parkinson's Disease is a neurodegenerative disorder that affects motor skills and speech. This project uses machine learning to analyze voice measurements and classify individuals for potential early detection. The implementation focuses on the Random Forest algorithm with hyperparameter tuning via GridSearchCV for optimized performance.

Dataset
Source: UCI Machine Learning Repository
Description: The dataset includes 195 records with 24 attributes, capturing various biomedical voice measurements and a status field (0 = healthy, 1 = Parkinson's Disease).
Features
Key features in the dataset include:

MDVP:Fo(Hz): Average vocal fundamental frequency
MDVP:Fhi(Hz): Maximum vocal fundamental frequency
MDVP:Flo(Hz): Minimum vocal fundamental frequency
HNR: Harmonic-to-Noise Ratio
PPE: Pitch Period Entropy
Status: Target variable indicating Parkinson's Disease presence (1) or absence (0)
Model and Techniques
Data Preprocessing:

Resampling using RandomOverSampler to address class imbalance.
Feature scaling with StandardScaler.
Model Training:

Random Forest Classifier with hyperparameter tuning via GridSearchCV.
Evaluation Metrics:

Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
ROC Curve and AUC Score
Results
Accuracy: 96.63%
ROC AUC Score: 0.99
Classification Report:
markdown
Copy code
             precision    recall  f1-score   support

         0       0.93      1.00      0.96        39
         1       1.00      0.94      0.97        50

  accuracy                           0.97        89
 macro avg       0.96      0.97      0.97        89
weighted avg 0.97 0.97 0.97 89

yaml
Copy code

---

## Dependencies

Install the required libraries using:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/Detecting-Parkinsons-Disease.git
cd Detecting-Parkinsons-Disease
Launch the notebook:

bash
Copy code
jupyter notebook
Run the Detecting Parkinsons Disease.ipynb notebook.

Contributing
Contributions are welcome! Please feel free to fork the repository and submit pull requests for any improvements or suggestions.

License
This project is licensed under the MIT License. See the LICENSE file for details.

