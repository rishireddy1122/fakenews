# Fake News Detector

A machine learning web app that predicts whether a news article is real or fake. It uses TF-IDF, metadata features, and Logistic Regression. The app is deployed on Streamlit Cloud and supports both single article input and CSV batch predictions.

---

## Features

* Single article prediction by pasting text.
* CSV upload for multiple articles at once.
* Shows prediction confidence.
* End-to-end pipeline with preprocessing and Logistic Regression.

---

## Live Demo

[Try the app here]  https://fakenewsprediction-k8vbzukmxs4bowsjpg7ndq.streamlit.app/

---

## Tech Stack

* Python
* scikit-learn (TF-IDF, Logistic Regression, GridSearchCV)
* Streamlit (UI and deployment)
* pandas, numpy (data preprocessing)
* joblib (model saving and loading)

---

## Project Structure

```
fake-news-detector/
│
├── data.csv                 # Dataset (optional)
├── train_model.py           # Training script
├── fake_news_pipeline.pkl   # Trained model pipeline
├── app.py                   # Streamlit app
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## Installation & Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

---

## Training the Model

Run the training script to build and save the pipeline:

```bash
python train_model.py
```

This will create `fake_news_pipeline.pkl`.

---

## Running the App Locally

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deployment on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and click "Deploy App".
3. Select `app.py` as the entry point.
4. The app will go live.

---

## Example Predictions

**Single Article**
Input: "NASA confirms discovery of water on Mars"
Output: Real News (Confidence: 92.5%)

**CSV Upload**

| Article Text                              | Prediction | Confidence |
| ----------------------------------------- | ---------- | ---------- |
| COVID vaccine turns humans into robots    | Fake       | 88.3%      |
| Apple releases iPhone 16 with AI features | Real       | 95.1%      |

---

## Skills Demonstrated

* Machine learning: preprocessing, feature engineering, Logistic Regression, hyperparameter tuning
* Deployment: building and deploying ML apps on Streamlit Cloud
* Data handling: single input and batch CSV predictions
* Model persistence with joblib

