SmartRoad – Traffic Accident Analytics

SmartRoad is a Computational Analytics and Machine Learning project that analyses UK road collision data to identify accident patterns, high-risk locations, and factors associated with serious or fatal collisions.

The project combines data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning, geographic hotspot clustering, and interactive data visualisation into a single web-based analytical application.

A user-friendly Streamlit dashboard allows users to explore collision patterns, investigate geographic hotspots, view analytical visualisations, and estimate the likelihood that a collision belongs to the Fatal or Serious severity category.

---

Live Application

The SmartRoad dashboard is publicly deployed using Streamlit Community Cloud:

https://smartroad-traffic-accident-analytics.streamlit.app

---

Project Objectives

The main objectives of SmartRoad are to:

1. Analyse UK road collision data to identify important accident patterns.
2. Understand the factors associated with serious and fatal collisions.
3. Identify geographic areas with concentrations of road collisions.
4. Perform systematic data cleaning and preprocessing.
5. Engineer useful features from temporal, road, environmental, and traffic variables.
6. Compare multiple classification algorithms for collision-severity prediction.
7. Evaluate machine-learning models using appropriate classification metrics.
8. Develop an interactive dashboard for data exploration and prediction.
9. Provide an accessible analytical tool for understanding road-safety data.

---

Key Features

- Official 2025 UK Department for Transport (DfT) STATS19 collision dataset
- Data cleaning, validation, and missing-value handling
- Feature engineering for:
  - Time and date
  - Road characteristics
  - Environmental conditions
  - Traffic conditions
  - Location
- Exploratory Data Analysis with eight visualisations
- Comparison of three machine-learning classification models:
  - Random Forest
  - Logistic Regression
  - Gradient Boosting
- 80/20 stratified train-test split
- 5-fold stratified cross-validation
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- K-Means geographic hotspot clustering
- Interactive Streamlit dashboard
- Collision-severity prediction form
- Live weather information through the Open-Meteo API
- Automated PDF report generation
- Public deployment through Streamlit Community Cloud
- Synthetic demo dataset for local software testing

---

Dataset

SmartRoad uses the official 2025 UK Department for Transport STATS19 road collision dataset.

The final analysis contains:

101,525 recorded collisions

The dataset provides information about road collisions, including variables relating to location, time, road conditions, environmental conditions, vehicles, casualties, and collision severity.

Target Variable

The machine-learning task is a binary classification problem:

Value| Meaning
"1"| Fatal or Serious
"0"| Slight

The objective of the prediction model is therefore to estimate whether a collision falls into the Fatal or Serious category rather than the Slight category.

Official Data Source

UK Department for Transport – Road Safety Open Data:

https://www.gov.uk/government/statistical-data-sets/road-safety-open-data

Demo Dataset

The repository also contains:

"demo_accidents.csv"

This is a small synthetic dataset intended for local software testing and demonstration. It is not used as the primary dataset for the reported 2025 model results.

---

Analysis Pipeline

The complete SmartRoad analytical workflow follows these stages:

                Data Acquisition
                       │
                       ▼
             Data Cleaning & Validation
                       │
                       ▼
              Exploratory Data Analysis
                       │
                       ▼
                Feature Engineering
                       │
                       ▼
             Train/Test Data Splitting
                       │
                       ▼
             Stratified Cross-Validation
                       │
                       ▼
                Model Training
                       │
                       ▼
                Model Comparison
                       │
                       ▼
             Performance Evaluation
                       │
                       ▼
              Geographic Clustering
                       │
                       ▼
             Interactive Dashboard
                       │
                       ▼
             Severity Risk Prediction

---

Data Preprocessing

Before analysis and machine learning, the raw collision data is processed through a structured preprocessing pipeline.

The preprocessing stage includes:

- Data type validation
- Duplicate detection
- Missing-value analysis
- Missing-value handling
- Invalid-value detection
- Categorical-variable processing
- Numerical-variable processing
- Data consistency checks
- Feature selection
- Preparation of model-ready data

This ensures that the data used for statistical analysis and machine learning is consistent and suitable for modelling.

---

Feature Engineering

Additional analytical features are created from the original STATS19 variables.

The feature-engineering process focuses on several categories:

Temporal Features

Examples include:

- Year
- Month
- Day
- Day of week
- Hour
- Time-of-day categories

These features help identify patterns such as differences between daytime and nighttime collisions or weekday and weekend accidents.

Road Features

Road-related information is transformed into useful analytical variables representing characteristics such as:

- Road type
- Speed limit
- Junction characteristics
- Road surface or condition
- Road infrastructure

Environmental Features

Environmental conditions are incorporated to investigate the relationship between collisions and factors such as:

- Weather
- Lighting conditions
- Road surface conditions
- Visibility-related conditions

Traffic and Location Features

Relevant traffic and geographic information is also incorporated to support:

- Collision-pattern analysis
- Geographic hotspot detection
- Machine-learning prediction

---

Exploratory Data Analysis

SmartRoad performs exploratory data analysis to understand the distribution and relationships within the collision dataset.

The project includes eight analytical visualisations covering areas such as:

- Collision severity distribution
- Collision trends over time
- Time-of-day accident patterns
- Day-of-week patterns
- Road-related factors
- Environmental conditions
- Geographic accident patterns
- Factors associated with serious or fatal collisions

The EDA stage is used to identify meaningful patterns before applying machine-learning algorithms.

---

Machine Learning

SmartRoad treats collision severity prediction as a binary classification problem.

Three classification algorithms were trained and evaluated using the same preprocessing and feature-engineering pipeline.

Models Used

1. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive performance and reduce overfitting.

2. Logistic Regression

Logistic Regression provides a comparatively simple and interpretable classification approach and is useful as a baseline model.

3. Gradient Boosting

Gradient Boosting builds an ensemble of models sequentially, with later models focusing on correcting errors made by earlier models.

---

Model Validation

To make the model comparison consistent and reliable, SmartRoad uses:

Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

A stratified split is used so that the class distribution is preserved between the training and testing sets.

Cross-Validation

The training data is evaluated using:

5-fold Stratified Cross-Validation

This provides a more robust estimate of model performance and reduces dependence on a single train-test split.

---

Model Evaluation Metrics

The models are evaluated using several classification metrics.

Accuracy

Measures the proportion of total predictions that are correct.

Precision

Measures how many observations predicted as Fatal or Serious are actually Fatal or Serious.

Recall

Measures how many actual Fatal or Serious collisions are successfully identified.

F1-score

The harmonic mean of precision and recall.

Confusion Matrix

Provides a detailed view of:

- True Positives
- True Negatives
- False Positives
- False Negatives

Because identifying serious collisions is important, recall and F1-score are considered alongside accuracy rather than relying on accuracy alone.

---

2025 Model Results

The three classification models were compared using the same preprocessing and feature-engineering pipeline.

Model| Accuracy| Precision| Recall| F1| CV F1
Random Forest| 65.01%| 33.85%| 34.92%| 34.38%| 34.59%
Logistic Regression| 57.33%| 32.60%| 58.66%| 41.91%| 41.85%
Gradient Boosting| 60.58%| 34.12%| 53.95%| 41.80%| 42.09%

---

Selected Model – Gradient Boosting

Gradient Boosting was selected as the final model because it achieved the highest mean 5-fold cross-validated F1-score among the evaluated models.

Final Performance

Metric| Result
Accuracy| 60.58%
Precision| 34.12%
Recall| 53.95%
F1-score| 41.80%
5-fold CV F1| 42.09% ± 0.24%
Records analysed| 101,525

The selected model is used by the dashboard for collision-severity prediction.

«Important: The prediction is an analytical risk estimate and should not be interpreted as a guarantee that a particular collision will be Fatal, Serious, or Slight.»

---

Geographic Hotspot Analysis

SmartRoad uses K-Means clustering to identify geographic concentrations of road collisions.

The clustering process analyses collision-location information to group geographically similar collision points.

This helps identify:

- Areas with high concentrations of collisions
- Geographic accident patterns
- Potential road-safety hotspots
- Spatial differences in collision activity

The results can be explored through the interactive dashboard.

---

Interactive Streamlit Dashboard

The SmartRoad dashboard provides an interactive interface for exploring the analytical results.

Dashboard capabilities include:

- Overview of the collision dataset
- Interactive accident statistics
- Exploratory visualisations
- Collision-severity analysis
- Geographic hotspot analysis
- Machine-learning model information
- Collision-severity prediction
- Live weather information
- Automated PDF reporting

Users can interact with analytical controls rather than relying only on static charts.

---

Collision Severity Prediction

The dashboard includes a user-friendly prediction form.

Users can provide relevant collision characteristics, after which the trained Gradient Boosting model estimates the predicted severity class.

The output is classified as either:

Fatal or Serious
        OR
Slight

The prediction functionality demonstrates how a trained machine-learning model can be integrated into an interactive analytical application.

---

Live Weather Integration

SmartRoad integrates live weather information using the Open-Meteo API.

Weather information can provide additional environmental context when interpreting road-safety conditions.

The API integration demonstrates how external real-time information can be incorporated into an analytical dashboard.

---

Automated PDF Reports

SmartRoad provides automated PDF report generation.

The generated report can be used to present analytical results in a structured document format, making it easier to share or review the findings outside the Streamlit dashboard.

---

Technology Stack

The project uses the following technologies:

Technology| Purpose
Python| Core programming language
Pandas| Data manipulation and analysis
NumPy| Numerical computation
Scikit-learn| Machine learning and clustering
Matplotlib| Data visualisation
Seaborn| Statistical visualisation
Streamlit| Interactive web dashboard
Open-Meteo API| Live weather information
ReportLab| PDF report generation
Git/GitHub| Version control and source-code management
Streamlit Community Cloud| Public deployment

---

Project Structure

A typical repository structure is:

SmartRoad/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── demo_accidents.csv
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── src/
│   └── ...
│
├── reports/
│   └── ...
│
└── assets/
    └── ...

«The exact file and folder names may vary depending on the final repository structure.»

---

Installation

1. Clone the Repository

git clone <repository-url>
cd SmartRoad

2. Create a Virtual Environment

python -m venv venv

Windows

venv\Scripts\activate

macOS/Linux

source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

---

Running the Dashboard Locally

Start the Streamlit application with:

streamlit run app.py

After starting the application, Streamlit will provide a local URL in the terminal, typically:

http://localhost:8501

Open the URL in a web browser to access the dashboard.

---

Reproducibility

The project uses a consistent preprocessing and feature-engineering pipeline across the evaluated machine-learning models.

The modelling workflow includes:

1. Data loading
2. Data validation
3. Data cleaning
4. Missing-value handling
5. Feature engineering
6. Feature preprocessing
7. Stratified 80/20 train-test split
8. 5-fold stratified cross-validation
9. Model training
10. Model evaluation
11. Model comparison
12. Selection of the final model

This structure helps ensure that model comparisons are performed under consistent conditions.

---

Limitations

SmartRoad is an analytical and educational project and has several limitations.

Data Limitations

The analysis is based on recorded road collisions. It therefore represents reported collision events rather than every road-safety incident that may have occurred.

Class Imbalance

Serious and fatal collisions represent a smaller proportion of the dataset than slight collisions. This affects classification performance and makes accuracy alone an insufficient evaluation metric.

Prediction Limitations

The machine-learning model provides a statistical prediction based on the available input variables. It should not be treated as a deterministic prediction or professional road-safety assessment.

Geographic Limitations

K-Means clustering identifies spatial groupings in the available collision data but does not by itself establish that a location is inherently dangerous.

Weather Information

Live weather information is obtained from an external API and may differ from the exact conditions present at the time and location of a historical collision.

---

Future Improvements

Potential future developments include:

- Hyperparameter optimisation
- Additional machine-learning algorithms
- Advanced class-imbalance techniques
- Explainable AI using SHAP or similar methods
- More detailed spatial analysis
- Interactive maps with additional geographic layers
- Temporal forecasting of collision trends
- More detailed weather integration
- Model probability calibration
- Automated model retraining with newer datasets
- Additional dashboard filters
- Improved PDF reporting
- Integration of road-network and traffic-volume data

---

Ethical and Practical Considerations

The purpose of SmartRoad is to support data-driven road-safety analysis.

Machine-learning predictions should be interpreted carefully because historical collision data can contain reporting limitations, geographic biases, and other sources of uncertainty.

The system should therefore be considered an analytical decision-support tool, rather than a replacement for professional road-safety investigation or emergency decision-making.

---

Project Outcome

SmartRoad demonstrates an end-to-end Computational Analytics workflow:

Raw Road Collision Data
          ↓
Data Cleaning
          ↓
Data Validation
          ↓
Feature Engineering
          ↓
Exploratory Data Analysis
          ↓
Machine Learning
          ↓
Model Evaluation
          ↓
Geographic Hotspot Detection
          ↓
Interactive Dashboard
          ↓
Collision Severity Prediction
          ↓
Automated Reporting

The project combines statistical analysis, machine learning, geographic clustering, API integration, interactive visualisation, and cloud deployment into a single road-accident analytics platform.

---

Conclusion

SmartRoad provides an end-to-end analytical framework for investigating UK road collisions and estimating collision severity.

Using 101,525 collision records from the 2025 UK DfT STATS19 dataset, the project performs data preprocessing, exploratory analysis, feature engineering, machine-learning model comparison, geographic hotspot clustering, and interactive visualisation.

Among the three evaluated classification models, Gradient Boosting achieved the highest mean 5-fold cross-validated F1-score of 42.09% ± 0.24% and was therefore selected for the final prediction component.

The resulting Streamlit application transforms the analytical workflow into an accessible interactive dashboard that can be used to explore collision patterns, investigate geographic hotspots, examine model performance, and generate collision-severity predictions.

---

Data Source

UK Department for Transport – Road Safety Open Data

https://www.gov.uk/government/statistical-data-sets/road-safety-open-data

Live Dashboard

SmartRoad – Traffic Accident Analytics

https://smartroad-traffic-accident-analytics.streamlit.app

---

License

This project is intended for academic and educational purposes.

The underlying road-safety data is provided by the UK Department for Transport and is subject to the terms and conditions associated with the original data source.

---

Author

SmartRoad – Traffic Accident Analytics

A Computational Analytics and Machine Learning project focused on understanding road collision patterns, geographic hotspots, and collision severity risk.