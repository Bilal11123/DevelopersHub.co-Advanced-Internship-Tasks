# Customer Churn Prediction Pipeline

## Objective
This project implements a reusable, production-ready machine learning pipeline for predicting customer churn using the Telco Churn dataset. The main objectives are:

- Build an end-to-end ML pipeline using scikit-learn's Pipeline API
- Implement proper data preprocessing (scaling, encoding)
- Train and compare multiple classification models
- Optimize model performance through hyperparameter tuning
- Package the complete pipeline for easy deployment

## Dataset
The Telco Customer Churn dataset contains information about telecom customers and whether they left the service (churned). Key features include:

- Demographic information (gender, senior citizen status)
- Account information (tenure, contract type)
- Services subscribed (phone, internet service add-ons)
- Billing information (monthly charges, payment method)

## Methodology/Approach

### 1. Data Preparation
- Load and clean data (handle missing values, convert data types)
- Separate features and target variable ('Churn')
- Stratified train-test split (80-20) to maintain class distribution

### 2. Preprocessing Pipeline
- **Numerical features**: Standard scaling (tenure, MonthlyCharges, TotalCharges)
- **Categorical features**: One-hot encoding (19 categorical columns)
- Implemented using `ColumnTransformer` for column-specific transformations

### 3. Model Training
- Implemented two classification models:
  - Logistic Regression (with L2 regularization)
  - Random Forest Classifier
- Created reusable pipeline combining preprocessing and model

### 4. Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation
- Different parameter grids for each model:
  - Logistic Regression: C, penalty, solver
  - Random Forest: n_estimators, max_depth, min_samples_split

### 5. Evaluation & Deployment
- Evaluated on test set using accuracy and classification report
- Saved complete pipeline (preprocessing + best model) using joblib

## Key Results/Observations

### Model Performance
| Model              | Test Accuracy | Precision (Churn) | Recall (Churn) |
|--------------------|---------------|-------------------|----------------|
| Logistic Regression| 0.780         | 0.79              | 0.78           |
| Random Forest      | 0.780         | 0.79              | 0.78           |

### Observations
1. **Class Imbalance**: The dataset has significant class imbalance (~73% non-churn, 27% churn)
2. **Model Comparison**:
   - Logistic Regression achieved slightly better overall accuracy
   - Random Forest showed better precision for churn class
3. **Feature Importance** (from Random Forest):
   - Contract type (month-to-month) was most predictive of churn
   - Tenure and monthly charges were also important factors
4. **Pipeline Benefits**:
   - The complete pipeline ensures consistent preprocessing for new data
   - Easy to update components (e.g., try different models or preprocessing)

## How to Use

### Requirements
- Python 3.7+
- scikit-learn
- pandas
- numpy
- joblib

### Running the Pipeline
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the pipeline: `python churn_pipeline.py`

### Using the Saved Model
```python
from joblib import load

# Load the pipeline
pipeline = load('churn_prediction_pipeline.joblib')

# Make predictions on new data
predictions = pipeline.predict(new_data)
