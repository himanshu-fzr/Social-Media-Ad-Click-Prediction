# Social Media Ad Click Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Predicts whether a user will click on a social media ad, using Logistic Regression — deployed as an interactive Streamlit web app.

## Overview

An end-to-end machine learning project: from synthetic data generation, through model training, to a deployed interactive web application. Users can input demographic and behavioral data and get a real-time click prediction with interpretable model insights (feature coefficients).

## Dataset

Generated a synthetic dataset of 30,000 user profiles using a logistic probability model, to control feature-target relationships and validate the full pipeline end-to-end before working with real ad-click data.

| Feature | Description |
|---|---|
| Age | 18–70 years |
| Gender | Male / Female |
| Income (kUSD) | Annual income, 20–200k |
| Time Spent (min) | Daily platform usage |
| Previous Clicks | Ad clicks in the last 30 days |
| Device Type | Desktop / Mobile |
| Ad Topic | Tech, Fashion, Sports, Food |
| Spending Score | 0–100 |
| Engagement Rate | 0–100 |
| Ad Frequency | Ads seen per day |
| **Click (Target)** | Whether the user clicked (Yes/No) |

## Approach

1. Generated and cleaned the synthetic dataset (`generate_ad_data.py`)
2. Scaled features using `StandardScaler`
3. Trained a Logistic Regression classifier (`train_ad_model.py`)
4. Evaluated using accuracy, precision, recall, and F1-score
5. Built an interactive Streamlit app (`app.py`) for real-time predictions, with feature-importance visualization and CSV export of results
6. Serialized the trained model and scaler with pickle for reuse in the app

## Results

| Metric | Score |
|---|---|
| Accuracy | **[ADD ACCURACY HERE]** |
| Precision | [ADD] |
| Recall | [ADD] |
| F1-score | [ADD] |

## Tech Stack

Python · Pandas · NumPy · Scikit-learn (Logistic Regression, StandardScaler) · Streamlit · Matplotlib · Seaborn

## Project Structure

```
Social-Media-Ad-Click-Prediction/
├── generate_ad_data.py   # Generate synthetic dataset (30k rows)
├── train_ad_model.py     # Train Logistic Regression & save model
├── app.py                # Streamlit web application
├── ad_click_data.csv     # Generated dataset
├── ad_model.pkl          # Trained model
├── ad_scaler.pkl         # Fitted StandardScaler
└── requirements.txt
```

## How to Run

```bash
git clone https://github.com/himanshu-fzr/Social-Media-Ad-Click-Prediction.git
cd Social-Media-Ad-Click-Prediction
pip install -r requirements.txt

python generate_ad_data.py    # generate the dataset
python train_ad_model.py      # train and save the model
streamlit run app.py          # launch the web app
```

## What I'd improve next

- Compare against tree-based models (Random Forest, XGBoost)
- Replace synthetic data with a real-world ad-click dataset (e.g. from Kaggle)
- Add SHAP-based explainability for individual predictions
- Deploy on Streamlit Cloud for a live public demo

## Contact

**Himanshu Sharma**
📧 himanshush0013@gmail.com · 🔗 [LinkedIn](https://www.linkedin.com/in/himanshusharmafzr) · 🐙 [GitHub](https://github.com/himanshu-fzr)
