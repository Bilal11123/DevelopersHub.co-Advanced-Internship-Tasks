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

---
## Evaluation
| Metric | Value (Example) |
| ------ | --------------- |
| MAE    | 0.04            |
| RMSE   | 0.06            |
