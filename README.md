# DevelopersHub.co-Advanced-Internship-Tasks
# News Topic Classification using BERT

## Objective
This project fine-tunes a BERT model to classify news headlines from the AG News dataset into four categories: World, Sports, Business, and Sci/Tech. The implementation focuses on achieving high accuracy while providing a complete pipeline from training to deployment.

## Methodology

### Implementation Details
1. **Dataset**: 
   - AG News dataset (120,000 training samples, 7,600 test samples)
   - Automatically loaded via Hugging Face's `datasets` library

2. **Model Architecture**:
   - Base model: `bert-base-uncased`
   - Added classification head with 4 output units
   - Default BERT weights initialized from pre-trained model

3. **Training Configuration**:
   - Batch size: 16 (per device)
   - Learning rate: 2e-5
   - Training epochs: 3
   - Weight decay: 0.01
   - Maximum sequence length: 128 tokens
   - Padding strategy: Fixed length padding

4. **Evaluation Metrics**:
   - Accuracy
   - Weighted F1-score

### Key Components
- **Tokenization**: Uses `BertTokenizerFast` for efficient processing
- **Training Loop**: Leverages Hugging Face's `Trainer` API
- **Model Saving**: Saves both model and tokenizer for deployment

## Results

### Performance Metrics
| Metric        | Value   |
|---------------|---------|
| Training Time | ~2 hours|
| Accuracy      | 94.8%   |
| F1-score      | 94.8%   |

### Training Observations
1. The model achieves excellent classification performance out of the box
2. Fixed-length padding simplifies implementation but may be less memory-efficient
3. Default hyperparameters work well for this task without extensive tuning
4. Three epochs proved sufficient for convergence

### Project Structure
Task 1/
1. ├── train.py             # Training script
2. ├── app.py               # Streamlit application
3. ├── ag_news_bert/        # Saved model directory
4. ├── README.md            # This file
5. └── requirements.txt     # Python dependencies

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

-----
# 🏡 Multimodal Housing Price Prediction

## 📌 Task 3: Multimodal ML – Predicting Housing Prices Using Images + Tabular Data

This project demonstrates a **multimodal machine learning** approach to predict housing prices by combining structured (tabular) features such as bedrooms and area with visual information (images of the houses). This mimics how real estate agents consider both numerical data and visuals when estimating property prices.

---

## 🎯 Objective

Build a deep learning model that:
- Processes **structured/tabular data** (e.g., number of rooms, area, location)
- Extracts features from **images** of houses
- Combines both data sources to predict **house prices**

---

## 🗂️ Dataset Structure

The dataset is obtained from the [Houses-dataset GitHub repository](https://github.com/emanhamed/Houses-dataset) and includes:
- A CSV file containing attributes and prices
- Folders with 4 images per house

The model has two branches:

## 🧾 Tabular Model (Structured Data)
Input: bedrooms, bathrooms, area, and zipcode
- Processed with:
- MinMaxScaler (continuous features)
- LabelBinarizer (zipcode)
- Model: Fully connected layers (Dense → Dense)

## 🖼️ Image Model (CNN)
- Input: A 64×64 RGB image (stitched from 4 house images)
- Model:
- 2 Convolutional + MaxPooling blocks
- Flatten → Dense(128)

## 🔗 Fusion Layer
- Concatenate both outputs
- Pass through Dense layers
- Final output: 1 unit (regression for price)

## Evaluation

| Metric | Value (Example) |
| ------ | --------------- |
| MAE    | 0.04            |
| RMSE   | 0.06            |

-----


# Auto Tagging Support Tickets with LLMs

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Transformers](https://img.shields.io/badge/🤗Transformers-4.30+-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A machine learning system that automatically categorizes customer support tickets using Large Language Models (LLMs) with three different approaches: zero-shot, few-shot, and fine-tuned classification.

## Features

- **Multi-method Approach**: Implements three distinct classification techniques
- **Multi-label Support**: Handles tickets with multiple relevant tags
- **Performance Comparison**: Evaluates different approaches on the same dataset
- **Production-ready**: Includes model saving/loading functionality

## Dataset

The system uses the [Customer Support Tickets dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) from Hugging Face:

- Contains 20k+ multi-language support tickets
- Each ticket has:
  - Subject line
  - Body text
  - 8 possible tags (tag_1 through tag_8)
