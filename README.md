# SmartRoad - Traffic Accident Analytics

SmartRoad is a Computational Analytics project for analysing road collisions, identifying high-risk patterns and geographic hotspots, comparing classification models, and estimating the probability that a collision is high severity (Fatal or Serious).

## What the project includes

- Official 2025 UK Department for Transport STATS19 collision dataset
- Data cleaning, missing-value handling and feature engineering
- Exploratory data analysis and eight visualisations
- Three classification models: Random Forest, Logistic Regression and Gradient Boosting
- Stratified train/test split and 5-fold cross-validation
- Accuracy, precision, recall, F1-score and confusion-matrix evaluation
- K-Means geographic hotspot clustering
- Interactive Streamlit dashboard with readable STATS19 field labels
- Live weather context through the Open-Meteo API
- Automated analytical PDF report generation

## 2025 Model Results

Gradient Boosting was selected using the highest mean 5-fold cross-validated F1-score.

- Accuracy: 60.58%
- Precision: 34.12%
- Recall: 53.95%
- F1-score: 41.80%
- 5-fold CV F1: 42.09% ± 0.24%
- Records: 101,525

## Data

The repository contains `data/road_collisions_2025.csv`, the 2025 collision dataset used for the project, together with a small synthetic `data/demo_accidents.csv` file for local software testing.

Official DfT source:
https://www.gov.uk/government/statistical-data-sets/road-safety-open-data

The analysis target is high-severity collision (Fatal or Serious) versus Slight. Outcome-derived variables that would leak the target are not used as predictive features.

## Run locally

Create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt
```

Run the automated tests:

```bash
python -m unittest discover -s tests -v
```

Run the analysis on the official 2025 dataset:

```bash
python run_analysis.py --data data/road_collisions_2025.csv
```

Start the dashboard:

```bash
streamlit run app.py
```

Generate the analytical PDF report:

```bash
python generate_report.py
```

## Live weather

The dashboard can retrieve current weather context for selected UK locations using the Open-Meteo API. These live values are displayed for context only and are not fed into the historical 2025 classifier.

## Repository structure

```text
├── app.py
├── download_data.py
├── generate_report.py
├── run_analysis.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── outputs/
├── notebooks/
├── src/
├── tests/
├── report/
└── presentation/
```
