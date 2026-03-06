# An Innovative Binary Model Framework for Cyberattack Detection and Classification in Imbalanced Domains

This repository contains the official implementation of the paper:
**"An Innovative Binary Model Framework for Cyberattack Detection and Classification in Imbalanced Domains"** by Óscar Mogollón-Gutiérrez, José Carlos Sancho Núñez, Mar Ávila, MohammadHossein Homaei, and Andrés Caro.

## Overview

Cyberattacks have increased in frequency and complexity, necessitating robust intrusion detection systems (IDS). This framework introduces an intelligent method for network traffic classification using machine learning and deep learning techniques. It specifically targets the pervasive issue of **class imbalance** in intrusion detection datasets by integrating the Synthetic Minority Oversampling Technique (SMOTE) within a One-vs-Rest (OvR) binary decomposition framework.

The framework operates in a structurally beneficial two-step process:
1. **Attack Detection (Step 1)**: A dedicated binary model filters normal, legitimate traffic from suspected anomalies.
2. **Attack Classification (Step 2)**: Suspected anomalies are classified into specific cyberattack categories by a multi-model ensemble system composed of specialized binary classifiers for each attack type.

## Key Contributions
- **Imbalance Mitigation**: Uses SMOTE during the generation of OvR binary datasets to equalize minority attack classes (e.g., U2R, R2L, Worms) without the noise typically associated with multiclass resampling.
- **Two-Step Architecture**: Effectively reduces the classification workload and mitigates hierarchical error propagation by filtering normal traffic first.
- **Algorithm Specialization**: Automatically selects the best-performing algorithm—from a pool of K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Trees (DT), and Multilayer Perceptron (MLP)—for each specific attack category.
- **Extensive Evaluation**: Validated across four established IDS datasets: NSL-KDD, UNSW-NB15, CSE-CICIDS2018, and ToN-IoT.

## Features
- **Data Preprocessing**: Handles missing values, performs one-hot encoding on categorical features, removes non-informative features, and applies standard scaling.
- **Binary Model Generation**: Follows an OvR methodology to generate balanced binary datasets using SMOTE for each attack category.
- **Model Training**: Evaluates K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Trees (DT), and Multilayer Perceptron (MLP) using Grid Search over 5-fold Cross-Validation, optimized for the Macro-F1 metric.
- **Two-Step Inference**: Combines the best binary models for a final prediction.
- **Imbalance-Aware Evaluation**: Focuses on Macro-F1, F1-Score per class, Precision, and Recall rather than just Accuracy.

## Repository Structure

```text
.
├── config.yaml               # Configuration file (paths to datasets, selected labels, models)
├── requirements.txt          # Python dependencies required to run the code
├── README.md                 # Project documentation
├── main.py                   # Main entry point for training and evaluation
└── src/
    ├── __init__.py
    ├── dataset.py            # Data loading, encoding, scaling, reducing features
    ├── model_generation.py   # Binary dataset generation and Grid Search CV
    ├── inference.py          # Two-step prediction inference framework
    └── evaluation.py         # Implements metrics: Confusion Matrix, Macro-F1, Precision, Recall
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd BinaryModels
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Modify the `config.yaml` file to set your dataset paths, model outputs, and general settings.
Ensure you point to the correct `.npy` or `.csv` files for your training and testing sets. Also, configure the `LABELS` according to the dataset you want to process (e.g., NSL-KDD, UNSW-NB15, CSE-CICIDS2018, ToN-IoT).

## Usage

To train the models and evaluate the framework, run:
```bash
python main.py
```
This script will execute the complete pipeline including data preprocessing, binary model generation, hyperparameter tuning, and metric evaluation.

## Results Summary (From the Paper)
The proposed binary framework significantly mitigates the impact of imbalance and improves the detection of minority attack profiles compared to standard multiclass approaches. Based on our evaluation:
- **NSL-KDD**: Achieved an F1-score of 0.7213.
- **UNSW-NB15**: Achieved an F1-score of 0.7754.
- **CSE-CICIDS2018**: Achieved an F1-score of 0.9340.
- **ToN-IoT**: Achieved an F1-score of 0.9793.

## Citation

If you use this codebase or find our work helpful, please cite our work:
```bibtex
@misc{Mogollon2026,
  author = {Óscar Mogollón-Gutiérrez and José Carlos Sancho Núñez and Mar Ávila and MohammadHossein Homaei and Andrés Caro},
  title = {An Innovative Binary Model Framework for Cyberattack Detection and Classification in Imbalanced Domains},
  journal = {CMC-Tech Science Press (Under Review)},
  year = {2026}
}
```
