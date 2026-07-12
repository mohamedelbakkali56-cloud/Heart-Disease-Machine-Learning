# Heart Disease Machine Learning

This project is a Machine Learning application developed in Python for predicting heart disease using supervised and unsupervised learning techniques. It includes data preprocessing, visualization, classification using Random Forest, clustering using K-Means, and dimensionality reduction with PCA.

## Features

- Data preprocessing and cleaning
- Handling missing values and duplicate records
- Feature encoding and scaling
- Random Forest Classification
- K-Means Clustering
- Principal Component Analysis (PCA)
- Correlation Heatmap
- Confusion Matrix
- Feature Importance Analysis
- Performance Evaluation (Accuracy, Precision, Recall, F1-Score)

## How to Use

### 1. Clone or download the project

```bash
git clone https://github.com/yourusername/Heart-Disease-Machine-Learning.git
```

### 2. Navigate to the project folder

```bash
cd Heart-Disease-Machine-Learning
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python heart_disease.py
```

## Project Workflow

1. Load and prepare the Heart Disease dataset.
2. Clean the data by removing duplicates and filling missing values.
3. Encode categorical features and normalize numerical features.
4. Train a Random Forest classifier.
5. Evaluate the model using several performance metrics.
6. Apply K-Means clustering.
7. Visualize clusters using PCA.
8. Generate charts and evaluation plots.

## Machine Learning Algorithms

### Supervised Learning

- Random Forest Classifier

### Unsupervised Learning

- K-Means Clustering
- Principal Component Analysis (PCA)

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```
Heart-Disease-Machine-Learning/
│
├── heart_disease.py
├── requirements.txt
├── README.md
└── images/
    ├── figure1.png
    ├── figure2.png
    ├── figure3.png
    ├── figure4.png
    └── figure5.png
```

## Future Enhancements

- Use the original UCI Heart Disease dataset.
- Compare multiple Machine Learning algorithms.
- Optimize hyperparameters using GridSearchCV.
- Build a web interface with Flask or Streamlit.
- Deploy the model online.

