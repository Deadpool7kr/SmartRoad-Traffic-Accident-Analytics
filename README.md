# SmartRoad - Traffic Accident Analytics

SmartRoad is a Computational Analytics project that analyses road collision data to identify accident patterns, high-risk locations and factors associated with serious collisions. It also provides a machine-learning based high-severity risk prediction through an interactive web dashboard.

## Project Features

- Official 2025 UK Department for Transport (DfT) STATS19 collision dataset
- Data cleaning, missing-value handling and validation
- Feature engineering for time, road, environmental and traffic-related variables
- Exploratory data analysis with eight visualisations
- Three classification models:
  - Random Forest
  - Logistic Regression
  - Gradient Boosting
- 80/20 stratified train-test split
- 5-fold stratified cross-validation
- Accuracy, precision, recall, F1-score and confusion-matrix evaluation
- K-Means geographic hotspot clustering
- Interactive Streamlit dashboard
- User-friendly collision severity prediction form
- Live weather information using the Open-Meteo API
- Automated PDF report generation
- Public web deployment through Streamlit Community Cloud

## 2025 Model Results

Three classification models were compared using the same preprocessing and feature-engineering pipeline.

Gradient Boosting was selected because it achieved the highest mean 5-fold cross-validated F1-score.

| Model | Accuracy | Precision | Recall | F1 | CV F1 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 65.01% | 33.85% | 34.92% | 34.38% | 34.59% |
| Logistic Regression | 57.33% | 32.60% | 58.66% | 41.91% | 41.85% |
| Gradient Boosting | 60.58% | 34.12% | 53.95% | 41.80% | 42.09% |

### Selected Model: Gradient Boosting

- Accuracy: **60.58%**
- Precision: **34.12%**
- Recall: **53.95%**
- F1-score: **41.80%**
- 5-fold CV F1: **42.09% ± 0.24%**
- Records analysed: **101,525**

The prediction target is a binary classification:

- **1 = Fatal or Serious**
- **0 = Slight**

## Dataset

The project uses the official 2025 UK Department for Transport STATS19 collision dataset.

The dataset contains **101,525 recorded collisions** used for the final analysis.

The repository also contains a small synthetic dataset (`demo_accidents.csv`) for local software testing.

Official source:

https://www.gov.uk/government/statistical-data-sets/road-safety-open-data

## Analysis Pipeline

```text
Data Acquisition
       ↓
Data Cleaning and Validation
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
5-Fold Cross-Validation
       ↓
Model Comparison
       ↓
Performance Evaluation
       ↓
Hotspot Clustering
       ↓
Streamlit Dashboard