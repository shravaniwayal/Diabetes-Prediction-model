# Diabetes Prediction using Logistic Regression
A machine learning-based classification project to predict whether a person is diabetic or not, based on medical diagnostic data. This project aims to support early diagnosis using logistic regression and fundamental data analysis techniques.

## Problem Statement
Diabetes is a growing health concern worldwide. Early detection is crucial to prevent complications such as heart disease, kidney failure, and vision loss. This project builds a machine learning model that can assist healthcare professionals in identifying high-risk individuals by analyzing health metrics such as glucose levels, BMI, and blood pressure.

## Potential Use Case
This model could be deployed by healthcare startups in rural India as a mobile-based screening tool. Health workers could collect patient data on-site and use the model to instantly predict diabetes risk — even in low-resource settings with limited internet or medical infrastructure.

## Dataset Overview
- **Name**: Pima Indians Diabetes Database
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Description**: This dataset includes several medical predictor variables and one target variable (`Outcome`). All patients are females of at least 21 years of age of Pima Indian heritage.
- **Records**: 768

## Features

| Feature          | Description                                 |
| ---------------- | ------------------------------------------- |
| Pregnancies      | Number of pregnancies                       |
| Glucose          | Plasma glucose concentration                |
| BloodPressure    | Diastolic blood pressure (mm Hg)            |
| SkinThickness    | Triceps skin fold thickness (mm)            |
| Insulin          | 2-Hour serum insulin (mu U/ml)              |
| BMI              | Body mass index                             |
| DiabetesPedigree | Diabetes pedigree function (family history) |
| Age              | Age in years                                |
| Outcome          | 0 = Non-diabetic, 1 = Diabetic              |

## Exploratory Data Analysis (EDA) Highlights
-Strong correlation observed between glucose levels, BMI, and diabetic outcome.

-Several columns (like insulin, skin thickness) contained zero values which were treated as missing and imputed using median.

-Diabetic patients tend to have higher BMI, more pregnancies, and higher glucose levels.

## Feature Engineering
-Handled missing values (zeroes) by replacing them with column medians.

-Scaled features using StandardScaler to improve model performance.

-No feature was dropped — logistic regression performed best with all predictors.

## Model Overview
- Algorithm: Logistic Regression

- Toolkits: Python, Scikit-learn, Pandas, Matplotlib, Seaborn

- Train/Test Split: 80/20

## Model Evaluation

| Metric        | Value |
| ------------- | ----- |
| Accuracy      | 77.9% |
| Precision     | 77.4% |
| Recall        | 62.2% |
| F1 Score      | 69.0% |
| ROC AUC Score | 84.7% |

### Confusion Matrix
- [[82 17]
  [21 34]]
 <img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/1882bd33-d1af-410c-9e78-d02cd936d38e" />

### Classification Report
```
                precision    recall  f1-score   support

           0       0.80      0.83      0.81        99
           1       0.67      0.62      0.64        55
    accuracy                           0.75       154
   macro avg       0.73      0.72      0.73       154
weighted avg       0.75      0.75      0.75       154
```
### ROC Curve and AUC Score
- AUC Score:  0.8229568411386594
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/6d21f3ab-eb65-4eaf-8f4f-02073179238e" />

### Model Interpretation
The model performs well overall but shows slightly lower recall for diabetic cases. In a real-world medical setting, false negatives (undetected diabetic cases) are riskier and need to be minimized. A future version could adjust the classification threshold or apply recall-optimized models.

## Future Enhancements
-Test other models like Random Forest, Decision Tree, or XGBoost

-Hyperparameter tuning using GridSearchCV

-Use SHAP or LIME for model explainability

-Build a Streamlit app for interactive predictions

-Integrate real-time form inputs for live testing

## Dependencies
Make sure you have the following Python libraries installed:

-pandas

-numpy

-matplotlib

-seaborn

-scikit-learn

-jupyter

## Get Started
Open the notebook in Jupyter:
     
      jupyter notebook diabetes_logistic.ipynb

## Notes
-Zero values in columns like BMI, BloodPressure, and Glucose were treated as missing and handled appropriately.

-Evaluation is based on one train/test split; performance may vary with cross-validation or different splits.

-The model can be extended to more complex classifiers like Random Forests or XGBoost.

## Project Structure
```
Diabetes-Prediction-model/
├── data/
│   └── diabetes.csv
├── diabetes_logistic.ipynb
└── README.md
```

## Author
**Shravani Wayal**

Data Science & ML Enthusiast | Python | Analytics
email: wayalshravani04@gmail.com
