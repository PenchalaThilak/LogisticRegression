# Logistic Regression on Adult Income Dataset

This project uses logistic regression to predict whether a person earns more than $50K per year based on census data from the UCI Adult dataset.

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Records**: ~32,000 entries
- **Features**: Age, workclass, education, marital status, occupation, race, sex, hours per week, etc.

## 🧠 Objective

To classify individuals into two income categories:
- `<=50K`
- `>50K`

Using logistic regression, we aim to predict this income class based on various demographic attributes.

## 🔧 Steps Performed

1. **Import Libraries**  
   - pandas, numpy, matplotlib, seaborn, sklearn

2. **Data Preprocessing**  
   - Load CSV
   - Handle missing values
   - Encode categorical variables

3. **Data Splitting**  
   - 80% training  
   - 20% testing

4. **Model Training**  
   - Logistic Regression from `sklearn`

5. **Evaluation Metrics**  
   - Accuracy
   - Classification report
   - Confusion matrix

## 🚀 Model Accuracy

The logistic regression model achieved an accuracy of around **82%** using all features.

## 📈 Visualizations

- Count plot for income distribution
- Age histogram
- Education level bar chart
- Confusion matrix heatmap

## 📂 Files

- `logistic_regression_adult.ipynb` — Jupyter Notebook with full code
- `README.md` — Project documentation

## 📌 Requirements

Install required libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
