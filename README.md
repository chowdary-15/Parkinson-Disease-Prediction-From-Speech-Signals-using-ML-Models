# 🧠 Parkinson's Disease Detection via Voice Analysis

A machine learning web application that detects early signs of **Parkinson's Disease** by analyzing vocal biomarkers extracted from a short voice recording. The user records themselves saying **"aaah"** for 5 seconds, and the app extracts 22 acoustic features to classify whether Parkinson's indicators are present.

---

## ✨ Features

- 🎙️ **Live voice recording** — 5-second in-browser recording via microphone
- 📐 **22 biomedical voice features** extracted automatically (jitter, shimmer, HNR, RPDE, DFA, D2, PPE, and more)
- 🤖 **Gradient Boosting Classifier** — best-performing model after comparing Random Forest, SVM, Logistic Regression, and Ensemble
- 📊 **Interactive visualizations** — waveform plot, spectrogram, radar chart, and feature metrics dashboard (Plotly)
- 🗃️ **Session history** — all recordings saved to a local SQLite database with timestamps, confidence scores, and editable notes
- 🌑 **Dark-themed UI** — built with Streamlit and custom CSS

---

## 🗂️ Project Structure

```
parkinsons-voice-detection/
│
├── gui.py                          # Main Streamlit application
├── Untitled.ipynb                  # Full ML pipeline: EDA → training → evaluation → export
├── parkinsons_disease.csv          # Dataset (195 samples, 22 features + target)
├── gradient_boosting_model.joblib  # Saved trained Gradient Boosting model
├── requirements.txt                # All Python dependencies
├── .gitignore                      # Files excluded from Git
└── README.md                       # This file
```

> ⚠️ The following files are **generated at runtime** — do not commit them:
> - `voice_analysis.db` — SQLite database, auto-created on first run
> - `temp_recording.wav`, `capture.wav`, `sample.wav` — temporary audio files

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/parkinsons-voice-detection.git
cd parkinsons-voice-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run gui.py
```

A browser tab will open at `http://localhost:8501`.

---

## 🎤 How to Use

1. Open the app in your browser
2. Click **"Start Recording"**
3. Wait for the **3-second countdown**
4. Say **"aaah"** clearly and steadily for **5 seconds**
5. View your **result** (Parkinson's Detected / Not Detected) with a confidence score
6. Explore the **waveform, spectrogram, radar chart, and feature breakdown**
7. Past recordings appear in the **sidebar** with key metrics and editable notes

> 💡 **Tips for best results:**
> - Use a quiet room
> - Position your mouth ~6 inches from the microphone
> - Maintain a steady, comfortable volume throughout

---

## 📊 Dataset

**Source:** [Parkinson's Disease Data Set – Kaggle (Vikalp Ukani)](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)

| Property | Value |
|---|---|
| Total samples | 195 voice recordings |
| Subjects | 31 people (23 with Parkinson's, 8 healthy) |
| Features | 22 biomedical voice measurements |
| Target column | `status` → 1 = Parkinson's, 0 = Healthy |
| Class imbalance | ~75% positive — handled with **SMOTE** |

### Feature Categories

| Category | Features |
|---|---|
| Fundamental Frequency | `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` |
| Jitter (frequency variation) | `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` |
| Shimmer (amplitude variation) | `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` |
| Noise Ratio | `NHR`, `HNR` |
| Nonlinear Dynamics | `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE` |

---

## 🧪 Model Training (`Untitled.ipynb`)

The notebook covers the full ML pipeline:

### 1. Exploratory Data Analysis
- Dataset shape, dtypes, missing values, duplicate check
- Target class distribution (count plot)
- Correlation heatmap across all 22 features
- Distribution plots and box plots per feature grouped by status
- Pair plots for jitter and shimmer feature groups

### 2. Preprocessing
- Dropped the `name` column (identifier, not a feature)
- Converted `status` to `uint8`
- Applied `LabelEncoder` for any remaining non-numeric columns
- 80/20 train-test split (`random_state=42`)
- Applied **SMOTE** on the training set to fix class imbalance

### 3. Models Trained & Compared

| Model | Pipeline |
|---|---|
| Random Forest | RobustScaler + SMOTE + RandomForestClassifier (500 trees) |
| SVM | StandardScaler + SMOTE + SVC (RBF / Poly kernel) |
| Logistic Regression | StandardScaler + SMOTE + LogisticRegression |
| **Gradient Boosting** ✅ | StandardScaler + SMOTE + GradientBoostingClassifier (500 trees) |
| Ensemble (Voting) | Soft-voting across all 4 best models |

- Hyperparameter tuning via **RandomizedSearchCV** (5-fold CV, 20 iterations, ROC-AUC scoring)
- Evaluation: Accuracy, ROC-AUC, Precision, Recall, F1, and train-test gap (overfitting score)

### 4. Model Selection & Export
**Gradient Boosting** was selected as the final model based on the best Test F1 score and stability ranking.

```python
import joblib
joblib.dump(best_models["Gradient Boosting"], 'gradient_boosting_model.joblib')
```

---

## 🏗️ App Architecture (`gui.py`)

### `ParkinsonsVoiceAnalyzer`
- Loads the `.joblib` model on startup
- `record_voice()` — captures 5 seconds of audio at 44100 Hz via `sounddevice`
- `extract_features()` — uses `librosa.pyin` for pitch (f0) extraction, then computes all 22 features
- `scale_features()` — applies min-max normalization using known feature ranges from the training data
- `predict()` — returns prediction (0/1) and class probabilities; uses a 98% confidence threshold for a positive Parkinson's result

### `DatabaseManager`
- SQLite with two tables: `recordings` (metadata + audio blob) and `features` (per-feature values per recording)
- Supports: save, retrieve all, delete single recording, clear all history

### `Visualizer`
- `create_waveform_plot()` — Plotly subplot showing waveform + spectrogram (dark theme)
- `create_feature_radar_plot()` — Z-score normalized radar chart of 6 key features
- `create_metrics_dashboard()` — 3-column layout: Voice Stability | Frequency Analysis | Advanced Metrics

### `display_history()` (Sidebar)
- All past recordings as expandable cards showing prediction, confidence, 4 key metrics, editable notes, and a delete button

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / Backend | Streamlit |
| Audio Recording | sounddevice, scipy |
| Audio Feature Extraction | librosa |
| ML Model | scikit-learn (Gradient Boosting) |
| Class Balancing | imbalanced-learn (SMOTE) |
| Visualization | Plotly |
| Database | SQLite |
| Model Serialization | joblib |
| Language | Python 3.8+ |

---

## ⚠️ Disclaimer

This application is a **research and educational project only**. It is **not a certified medical diagnostic tool** and should **not** be used as a substitute for professional medical evaluation. Always consult a licensed neurologist or healthcare provider for any medical concerns.