# 📱 Social Media Ad Click Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

> **Predict whether a user will click on a social media ad using Logistic Regression. Interactive web app built with Streamlit.**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results & Performance](#results--performance)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🔍 Overview

This project demonstrates an end-to-end **machine learning pipeline** for predicting ad clicks on social media platforms. Using a synthetic dataset of 30,000 user profiles, we train a **Logistic Regression** model to classify whether a user is likely to click on an ad based on demographic, behavioral, and content features.

The final output is an interactive **Streamlit web application** where users can input their own data and receive real-time predictions with interpretable insights.

---

## ✨ Features

- ✅ **Generate synthetic dataset** (30,000 samples) with realistic user features.
- ✅ **Train Logistic Regression** model with feature scaling.
- ✅ **Interactive web app** with sliders, radio buttons, and real-time prediction.
- ✅ **Interpretable ML** – Shows feature importance (coefficients) and model performance.
- ✅ **Download predictions** as CSV for further analysis.
- ✅ **Professional UI** with gradients, animations, and responsive layout.
- ✅ **No functions / No OOP** – Simple sequential code, perfect for beginners.

---

## 🛠️ Tech Stack

| Tool/Library       | Purpose                          |
|--------------------|----------------------------------|
| Python 3.8+        | Core programming language       |
| Pandas             | Data manipulation & storage      |
| NumPy              | Numerical operations             |
| Scikit-learn       | Logistic Regression, scaling, metrics |
| Streamlit          | Web application framework        |
| Matplotlib / Seaborn | Data visualization & charts   |

---

## 📊 Dataset Description

The dataset contains **30,000 synthetic user records** with the following features:

| Feature               | Type      | Description                                           |
|-----------------------|-----------|-------------------------------------------------------|
| Age                   | Integer   | 18–70 years                                           |
| Gender                | Binary    | 0 = Female, 1 = Male                                  |
| Income_kUSD           | Integer   | Annual income (20–200 kUSD)                          |
| TimeSpent_min         | Integer   | Minutes spent on platform per day (0–300)            |
| PrevClicks            | Integer   | Number of ad clicks in last 30 days (0–50)           |
| DeviceType            | Binary    | 0 = Desktop, 1 = Mobile                               |
| AdTopic               | Categorical | 0 = Tech, 1 = Fashion, 2 = Sports, 3 = Food       |
| SpendingScore         | Integer   | 0–100 (user purchasing tendency)                     |
| EngagementRate        | Integer   | 0–100 (user engagement with content)                 |
| AdFrequency           | Integer   | Ads seen per day (1–30)                              |
| **Click (Target)**    | Binary    | 0 = No click, 1 = Click                              |

> ⚠️ The dataset is **synthetic** and generated using a logistic probability model to ensure realistic relationships between features and the target.

---

## 📁 Project Structure

Social_Media_Ad_Click_Prediction/
│
├── generate_ad_data.py # Generate synthetic dataset (30k rows)
├── train_ad_model.py # Train logistic regression & save model
├── app.py # Streamlit web application
├── ad_click_data.csv # Generated dataset (after running step 1)
├── ad_model.pkl # Trained logistic regression model
├── ad_scaler.pkl # Fitted StandardScaler

---

## ⚙️ Installation

Follow these steps to run the project on your local machine.
### 1️⃣ Clone the repository

```
### bash
git clone https://github.com/yourusername/Social-Media-Ad-Click-Predictor.git
cd Social-Media-Ad-Click-Prediction2

----
## 🔮 Future Improvements
Add more advanced models (Random Forest, XGBoost) for comparison.

Integrate real-world ad click data (e.g., from Kaggle).

Add SHAP explainability plots.

Deploy the app on Streamlit Cloud.

Add user authentication to save prediction history.

---
📄 License
This project is licensed under the MIT License – you are free to use, modify, and distribute it for personal or commercial purposes.

---
