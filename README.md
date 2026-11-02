# ANN Customer Churn Prediction

## Overview

This project uses an Artificial Neural Network (ANN) to predict whether a bank customer is likely to leave the bank.

The project covers the complete machine learning pipeline, including data preprocessing, categorical encoding, feature scaling, ANN model construction, regularization, training, and evaluation.

## Dataset

The project uses the **Churn Modelling** dataset.

The dataset contains information about bank customers, including features such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card ownership
- Active Membership
- Estimated Salary

The target variable is **Exited**, which indicates whether the customer left the bank.

## Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

## Machine Learning Pipeline

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
One-Hot Encoding
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
ANN Model
     ↓
Dropout / Regularization
     ↓
Model Training
     ↓
Prediction
     ↓
Model Evaluation
```

## Data Preprocessing

### One-Hot Encoding

Categorical features such as `Geography` and `Gender` are converted into numerical representations using one-hot encoding.

### Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

### Feature Scaling

`StandardScaler` is used to standardize the numerical features before training the ANN.

Standardization follows:

\[
z = \frac{x-\mu}{\sigma}
\]

where:

- \(x\) = original feature value
- \(\mu\) = training-set mean
- \(\sigma\) = training-set standard deviation

## ANN Architecture

The model consists of:

```text
Input Layer
     ↓
Dense Layer + ReLU
     ↓
Dropout
     ↓
Dense Layer + ReLU
     ↓
Dropout
     ↓
Output Layer + Sigmoid
```

### ReLU

ReLU is used in the hidden layers to introduce non-linearity:

\[
ReLU(x)=\max(0,x)
\]

### Sigmoid

The output layer uses the sigmoid activation function because this is a binary classification problem.

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

The output can be interpreted as the probability that a customer will churn.

## Regularization

Dropout is used to reduce overfitting.

During training, a random fraction of neuron activations is temporarily set to zero. This prevents the network from becoming overly dependent on specific neurons and encourages better generalization.

## Model Training

The ANN is trained using batches over multiple epochs.

The training process uses:

- Binary classification
- Loss function
- Optimizer
- Validation data
- Dropout for regularization

## Model Evaluation

The trained model is evaluated on unseen test data.

Evaluation metrics include:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

## Files

| File | Description |
|---|---|
| `ann_churn_prediction.ipynb` | Complete data preprocessing, model training, and evaluation |
| `ann_churn_model.keras` | Saved trained ANN model |
| `Churn_Modelling.csv` | Dataset |
| `README.md` | Project documentation |

## How to Run

The notebook was developed using Google Colab.

1. Clone or download this repository.
2. Open `ann_churn_prediction.ipynb` in Google Colab.
3. Make sure the dataset is available in the expected location.
4. Run the notebook cells sequentially.

## Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Comparing ANN with traditional machine learning models
- Handling class imbalance if required
- Improving model architecture
- Deploying the model as a web application or API
