"""
Scientific Data Analysis Project: Predicting Diabetes with multiple lineair and logistic regression

Description: After selecting relevant parameters, selected based on sufficient correlation (A) 
or stepwise-model selection (B), linear (I) and logistic (II) regression models were constructed
to predict the development of diabetes based on analysed physiological and metabolic parameters.

Date of last modification: 13 December 2025

Authors:
Yael Ai, 14716674
Jelmer van Dam, 14802333
Glenn de Graaff, 1514913
Wong Zheng Tat, 16365453

Link to the original dataset of the study where the cleaned dataset was constructed from: https://datadryad.org/dataset/doi:10.5061/dryad.ft8750v
#citations
Study from which the original dataset came from: Chen, Y., Zhang, X. P., Yuan, J., Cai, B., Wang, X. L., Wu, X. L., Zhang, Y. H., Zhang, X. Y.,
Yin, T., Zhu, X. H., Gu, Y. J., Cui, S. W., Lu, Z. Q., & Li, X. Y. (2018). Association of body mass index and age with incident diabetes in 
Chinese adults: a population-based cohort study. BMJ open, 8(9), e021768. https://doi.org/10.1136/bmjopen-2018-021768
"""
## Section 0. Data Processing 
# The dataset used for the diabetes prediction models was constructed from the original dataset, after removing all empty entries (199230 rows) and unused parameters
# (3 columns: ID, length between first measurements and follow-up appointment took place, and the location of the measurements) in the programming language R.


## Section 1. Data description
# The cleaned dataset contains a sample of 12356 Chinese adults (age >20) for which various parameters were measured (see below) in a health check and a
# follow-up at least 2 years later where diabetes was monitored, both taking place in the period 2010-2016. The patients had no previous diagnosis of diabetes or 
# diabetes during the first measurements.

# Parameters, measured at first visit: 
# Age (years)
# Gender (1: male, 2: female)
# BMI (Body Mass Index) (kg/m2)
# SBP (Systolic Blood Pressure) (mmHg)
# DBP (Diastolic Blood Pressure) (mmHg)
# Cholesterol (mmol/L)
# Triglyceride (mmol/L)
# FPG (Fasting Plasma Glucose) (mmol/L)
# HDL (High-Density Lipoprotein) (mmol/L)
# LDL (Low-Density Lipoprotein) (mmol/L)
# ALT (Alanine Aminotransferase) (U/L)
# BUN (Blood Urea Nitrogen) (mmol/L)
# Smoking Status: (1: Current Smoker, 2: Ever Smoker, 3: Never Smoker)
# Drinking Status: (1: Current Drinker, 2: Ever Drinker, 3: Never Drinker)
# Family History of Diabetes: (1: Yes, 0: No)

# Parameters at final visit:
# FFPG (Fasting Plasma Glucose at Final Visit) (mmol/L)
# Diabetes (1: Yes, 0:No)


## Section 2. Data visualisation: looking for outliers.
# All Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
# Loading in dataset
df = pd.read_csv("diabetes_cleaned.csv")
df.columns = df.columns.str.strip()

print("Header: ")
print(df.head())
# We create the histogram of each feature to look for any statistical outliers.
cols = list(df.columns)
for col in cols:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f"Histogram of {col}.png")
# # No outliers were found
# Reset figure numbering for subsequent figures
figure_number = 1


## Section 3A. Feature selection based on correlations with FFPG
# We select the parameters that are sufficiently correlated with diabetes.
dependent_variable_name = "FFPG"
if dependent_variable_name == "Diabetes":
    X = df.drop("FFPG",axis=1) # Explanatory variables before feature selection
elif dependent_variable_name == "FFPG":
    X = df.drop("Diabetes",axis=1) 
y = df[dependent_variable_name] # The dependent variable
# Create a barplot of the correlations between the dependent variable and the explanatory variables
col_names = df.columns
corr_matrix = X.corr()
corrs = corr_matrix[dependent_variable_name]
corr_filtered = corrs.drop(dependent_variable_name)
corr_sorted = corr_filtered.sort_values()
# Plot the correlations
plt.figure(figsize=(10, 5)) 
plt.bar(corr_sorted.index, corr_sorted.values)
plt.xticks(rotation=45, ha="right")
plt.title(f"Figure {figure_number}: Correlation with {dependent_variable_name}")
plt.savefig(f"Figure {figure_number} - Correlation with {dependent_variable_name}.png")
figure_number += 1
# Feature Selection: select the variables that have Pearson's correlation |r| > 0.2 with the dependent variable.
selection = []
for col_name, corr in corrs.items():
    if abs(corr) > 0.2 and col_name not in ["FFPG", "Diabetes"]:
        selection.append(True)
    else:
        selection.append(False)
X_selection = X.loc[:, np.array(selection)]
X_corr = X_selection
selected_features = X_selection.columns
print(f"===== Selected features based on correlation with {dependent_variable_name} =====")
print(f"Selected features: {list(selected_features)}")
print(f"=========================================================\n")
X_selection= X_selection.to_numpy()
# Select the sufficiently correlated variables and create a plot of the dependent variable as a function of those variables
X_selection_names = list(selected_features)
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 10))
ax = ax.flatten()
plt.title(f"Figure {figure_number}: Correlation plots with FFPG")
for i in range(len(selected_features)):
    ax[i-1].scatter(df[selected_features[i]], y=df["FFPG"])
    ax[i-1].set_title(f"FFPG as function of {X_selection_names[i]}")
plt.savefig(f"Figure {figure_number} - Correlation plots with FFPG.png") 
# Note that in the plot of FFPG as function of FPG, some values for FPG were rounded to 3 in the original dataset.
figure_number += 1

# Create a copy of the dataset with BMI and Weight interaction term for correlation-selected features
df_corr = df.copy()
df_corr["BMI_Weight_interaction"] = df_corr["BMI"] * df_corr["weight"]

# Update X_corr to include the new interaction term
X_corr_updated = df_corr[list(selected_features) + ["BMI_Weight_interaction"]]
  
## Section 3B. Feature selection based on a stepwise selection
# Stepwise selection automatically choose good features for our model using mlxtend library.
# The SequentialFeatureSelector performs forward, backward, or bidirectional stepwise selection.
# It evaluates feature subsets using cross-validation and selects features that optimize model performance.
def stepwise_selection_wrapper(X_df, y, model, direction='forward', k_features='best', scoring='r2', cv=5, verbose=True):
    if k_features == 'best':
        k_features = (1, X_df.shape[1])
    
    # Set direction parameters
    if direction == 'bidirectional':
        forward = True
        floating = True

    sfs = SFS(model,
              k_features=k_features,
              forward=forward,
              floating=floating,
              scoring=scoring,
              cv=cv,
              n_jobs=-1,
              verbose=2 if verbose else 0)
    
    # Fit the selector
    sfs.fit(X_df.values, y.values)
    
    # Get selected features by index and map to column names
    selected_indices = list(sfs.k_feature_idx_)
    selected_features = [X_df.columns[i] for i in selected_indices]
    
    return selected_features

# Section 3B.I. Stepwise selection for OLS with as dependent variable FFPG
print("Stepwise selection for OLS with as dependent variable FFPG: ")
dependent_variable_name = "FFPG"
y = df[dependent_variable_name] 
X = df.drop("FFPG",axis=1).drop("Diabetes",axis=1)

best_features_stepwise_OLS_FFPG = stepwise_selection_wrapper(
    X, y, 
    model=LinearRegression(), 
    direction='bidirectional', 
    k_features='best',
    scoring='r2',
    cv=5
)
# Section 3B.II. Stepwise selection for logistic regression with as dependent variable Diabetes
dependent_variable_name = "Diabetes"
y = df[dependent_variable_name]
X = df.drop("FFPG",axis=1).drop("Diabetes",axis=1)

best_features_stepwise_Logit = stepwise_selection_wrapper(
    X, y, 
    model=LogisticRegression(max_iter=2000, class_weight="balanced"), 
    direction='bidirectional', 
    k_features='best',
    scoring='f1',
    cv=5
)
print("\n")
print(f"===== Selected features based on stepwise selection using OLS with as dependent variable {dependent_variable_name} =====")
print(f"Selected features: {list(best_features_stepwise_OLS_FFPG)}")
print(f"=========================================================\n")
print(f"===== Selected features based on stepwise selection using logistic regression with as dependent variable {dependent_variable_name} =====")
print(f"Selected features: {list(best_features_stepwise_Logit)}")
print(f"=========================================================\n")
## We have identified two methods for feature selection. We will proceed with both methods and compare which method is better.
## Section 4B. Checking for Interaction Terms
# Now, we will check for interaction terms in the features selected. We can check by using a correlation matrix.
# Each cell in the matrix represents the correlation between two variables. High positive or negative correlations suggest that the features 
# move together and may contain overlapping information. This helps identify potential interaction terms,
# multicollinearity, or features that may not add much new information to the model.
def create_plots_section_4(best_features_stepwise, dependent_variable_name, figure_number):

    new_df_stepwise = df[best_features_stepwise].copy()
    new_df_bestK = df[selected_features].copy()

    # Stepwise features heatmap
    plt.figure(figsize=(12, 10))
    corr = new_df_stepwise.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title(f"Figure {figure_number}: Correlation Matrix of Stepwise Selection Features")
    plt.savefig(f"Figure {figure_number} - Correlation Matrix Stepwise {dependent_variable_name}.png")
    figure_number += 1

    # Best_K features heatmap
    plt.figure(figsize=(12, 10))
    corr = new_df_bestK.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title(f"Figure {figure_number}: Correlation Matrix of Best_K Features")
    plt.savefig(f"Figure {figure_number} - Correlation Matrix Best_K Features.png")
    figure_number += 1

    return figure_number

# Section 4B.I. Check for interaction terms in OLS with dependent variable FFPG
figure_number = create_plots_section_4(best_features_stepwise=best_features_stepwise_OLS_FFPG, dependent_variable_name="FFPG", figure_number=figure_number)
# Section 4B.II. Check for interaction terms in logistic regression (using OLS-selected features)
figure_number = create_plots_section_4(best_features_stepwise=best_features_stepwise_Logit, dependent_variable_name="Diabetes", figure_number=figure_number)

# Create a copy of the dataset for logistic regression with interaction terms
df_logit = df.copy()

# Add interaction terms for logistic regression
interaction_pairs_logit = {
    "LDL_Cholesterol_interaction": ("LDL", "cholesterol"),
    "CCR_Gender_interaction": ("CCR", "gender"),
    "AST_ALT_interaction": ("AST", "ALT"),
}
for new_col, (col_a, col_b) in interaction_pairs_logit.items():
    df_logit[new_col] = df_logit[col_a] * df_logit[col_b]

# Create a copy of the dataset for linear regression with interaction terms
df_ols = df.copy()

# Add interaction terms for OLS linear regression
interaction_pairs_ols = {
    "Gender_Drinking_interaction": ("gender", "drinking"),
    "Weight_Gender_interaction": ("weight", "gender"),
    "DBP_SBP_interaction": ("DBP", "SBP"),
    "AST_ALT_interaction": ("AST", "ALT"),
}
for new_col, (col_a, col_b) in interaction_pairs_ols.items():
    df_ols[new_col] = df_ols[col_a] * df_ols[col_b]



## Section 5A.I. Predicting diabetes using linear regression on the features selected based on sufficient correlation
# Set FFPG as dependent variable
y_ffpg = df_corr['FFPG']
# Use the features that are sufficiently correlated with FFPG (with BMI×Weight interaction)
X = X_corr_updated 
# Split
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X, y_ffpg, test_size=0.2, random_state=42
)
# Scale
scaler_f = StandardScaler()
X_train_scaled_f = scaler_f.fit_transform(X_train_f)
X_test_scaled_f = scaler_f.transform(X_test_f)
# Train model
linreg_f = LinearRegression()
linreg_f.fit(X_train_scaled_f, y_train_f)
# Predict
y_pred_f = linreg_f.predict(X_test_scaled_f)
# Evaluation metrics
mse_f = mean_squared_error(y_test_f, y_pred_f)
r2_f = linreg_f.score(X_test_scaled_f, y_test_f)
# Calculate accuracy by classifying as diabetes if FFPG >= 7.0 mmol/L
y_test_binary_f = (y_test_f >= 7.0).astype(int)
y_pred_binary_f = (y_pred_f >= 7.0).astype(int)
accuracy_f = accuracy_score(y_test_binary_f, y_pred_binary_f)
conf_matrix_f = confusion_matrix(y_test_binary_f, y_pred_binary_f)
print("===== Linear Regression Results predicting FFPG with Correlation Selected Features =====")
print("MSE:", mse_f)
print("R²:", r2_f)
print("Accuracy (diabetes classification at FFPG >= 7.0):", accuracy_f)
print("Confusion Matrix (diabetes classification):\n", conf_matrix_f)
# Plot: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(np.arange(len(y_test_f)), y_test_f, color='black', alpha=0.6, label='Actual FFPG')
plt.scatter(np.arange(len(y_test_f)), y_pred_f, color='red', alpha=0.6, label='Predicted FFPG')
plt.title(f' Figure {figure_number}: Actual vs Predicted FFPG (Linear Regression)')
plt.xlabel('Sample Index')
plt.ylabel('FFPG Value')
plt.legend()
plt.savefig(f"Figure {figure_number} - FFPG_Actual_vs_Predicted.png")
figure_number += 1
## Section 5B.II. Predicting diabetes using logistic regression on the features selected based on sufficient correlation 
# Set Diabetes as dependent variable  
y_diabetes = df_corr['Diabetes']
# Build feature matrix for logistic regression using features that are sufficiently correlated with FFPG (with BMI×Weight interaction)
X_logit = X_corr_updated.values
X_train, X_test, y_train, y_test = train_test_split(X_logit, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg = LogisticRegression(max_iter=5000, class_weight="balanced")
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 5))
plt.scatter(np.arange(len(y_test)), y_test, color='black', label='Actual Values', alpha=0.6)
plt.scatter(np.arange(len(y_test)), y_pred, color='blue', label='Logistic Regression Predictions', alpha=0.6)
plt.title(f'Figure {figure_number}: Actual values vs Logistic Regression Predictions based on correlation-selected features')
plt.xlabel('Sample Index')
plt.ylabel('Diabetes (Actual) vs Predicted Value (Logistic Regression)')
plt.legend()
plt.savefig(f"Figure {figure_number} - Actual values vs Logistic Regression Predictions based on correlation-selected features.png")
figure_number += 1
print("\n===== Logistic Regression predicting Diabetes with Correlation Selected Features =====")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification report:")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))
print("=========================================================\n")

## Section 5B. Linear and Logistic Regression with Stepwise-Selected Features
# Section 5B.I. Linear Regression for FFPG using stepwise-selected features
# Build feature matrix using stepwise-selected features
X_stepwise = df_ols[best_features_stepwise_OLS_FFPG]
y_ffpg_stepwise = df_ols['FFPG']
# Split
X_train_sw, X_test_sw, y_train_sw, y_test_sw = train_test_split(
    X_stepwise, y_ffpg_stepwise, test_size=0.2, random_state=42
)
# Scale
scaler_sw = StandardScaler()
X_train_scaled_sw = scaler_sw.fit_transform(X_train_sw)
X_test_scaled_sw = scaler_sw.transform(X_test_sw)
# Train model
linreg_sw = LinearRegression()
linreg_sw.fit(X_train_scaled_sw, y_train_sw)
# Predict
y_pred_sw = linreg_sw.predict(X_test_scaled_sw)
# Evaluation metrics
mse_sw = mean_squared_error(y_test_sw, y_pred_sw)
r2_sw = linreg_sw.score(X_test_scaled_sw, y_test_sw)
# Calculate accuracy by classifying as diabetes if FFPG >= 7.0 mmol/L
y_test_binary_sw = (y_test_sw >= 7.0).astype(int)
y_pred_binary_sw = (y_pred_sw >= 7.0).astype(int)
accuracy_sw = accuracy_score(y_test_binary_sw, y_pred_binary_sw)
conf_matrix_sw = confusion_matrix(y_test_binary_sw, y_pred_binary_sw)
print("\n===== FFPG Linear Regression with Stepwise-Selected Features =====")
print("MSE:", mse_sw)
print("R²:", r2_sw)
print("Accuracy (diabetes classification at FFPG >= 7.0):", accuracy_sw)
print("Confusion Matrix (diabetes classification):\n", conf_matrix_sw)
print(f"=========================================================\n")
# Section 5B.II.: Logistic Regression for Diabetes using stepwise-selected features
# Build feature matrix using stepwise-selected features
y_diabetes_sw = df['Diabetes']
# Split (reuse X_stepwise with stepwise features)
X_train_log_sw, X_test_log_sw, y_train_log_sw, y_test_log_sw = train_test_split(
    df_logit, y_diabetes_sw, test_size=0.2, random_state=42, stratify=y_diabetes_sw
)
# Scale
scaler_log_sw = StandardScaler()
X_train_scaled_log_sw = scaler_log_sw.fit_transform(X_train_log_sw)
X_test_scaled_log_sw = scaler_log_sw.transform(X_test_log_sw)
# Train model
logreg_sw = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg_sw.fit(X_train_scaled_log_sw, y_train_log_sw)
# Predict
y_pred_log_sw = logreg_sw.predict(X_test_scaled_log_sw)
# Evaluation metrics
accuracy_log_sw = accuracy_score(y_test_log_sw, y_pred_log_sw)
conf_matrix_log_sw = confusion_matrix(y_test_log_sw, y_pred_log_sw)
print("\n===== Diabetes Logistic Regression with Stepwise-Selected Features =====")
print("Accuracy:", accuracy_log_sw)
print("Confusion Matrix:\n", conf_matrix_log_sw)
print("Classification report:")
print(classification_report(y_test_log_sw, y_pred_log_sw, digits=3, zero_division=0))
print("=========================================================\n")
# Plot actual vs predicted probabilities for the stepwise logistic regression
y_pred_prob_log_sw = logreg_sw.predict_proba(X_test_scaled_log_sw)[:, 1]
plt.figure(figsize=(8, 5))
plt.scatter(np.arange(len(y_test_log_sw)), y_test_log_sw, color='black', alpha=0.6, label='Actual')
plt.scatter(np.arange(len(y_test_log_sw)), y_pred_prob_log_sw, color='blue', alpha=0.6, label='Predicted probability')
plt.title(f'Figure {figure_number}: Actual vs Predicted Probability (Stepwise Logistic Regression)')
plt.xlabel('Sample Index')
plt.ylabel('Diabetes Probability')
plt.legend()
plt.savefig(f"Figure {figure_number} - Actual_vs_Predicted_Probability_Stepwise_LogReg.png")
figure_number += 1
# Comparing the accuracy and confusion matrix between linear and logistics regression, we can observe that they have comparable accuracy of about 98%. 
# We can analyse the results further by taking a look at a Predictions vs Actual Values Plot
# Train linear regression model for Diabetes (for comparison with logistic regression)
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
y_pred_linreg = linreg.predict(X_test_scaled)
y_pred_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
plt.figure(figsize=(12, 8))
plt.scatter(np.arange(len(y_test)), y_test, color='black', label='Actual Values', alpha=0.6)
plt.scatter(np.arange(len(y_test)), y_pred_linreg, color='red', label='Linear Regression Predictions', alpha=0.6)
plt.scatter(np.arange(len(y_test)), y_pred_logreg, color='blue', label='Logistic Regression Predictions', alpha=0.6)
plt.title(f'Figure {figure_number}: Linear Regression vs Logistic Regression Predictions vs Actual Values based on stepwise model selection')
plt.xlabel('Sample Index')
plt.ylabel('Predicted vs Actual Value (Diabetes: 0 or 1)')
plt.legend()
plt.savefig(f"Figure {figure_number} - Linear Regression vs Logistic Regression Predictions vs Actual Values based on stepwise model selection.png")
# From the analysis above, we can see that predicted values for linear regression exceeds the bound of reality for probability as compared to logistics regression. 
# On the other hand, logistics regression falls within the bound of 0 and 1, making it more suitable for calculating probability.
figure_number += 1
# 1. Inspect the Linear Regression predictions range
print("===== Linear Regression Predictions Range =====")
print(f"Min: {y_pred_linreg.min()}, Max: {y_pred_linreg.max()}")
print(f"=========================================================\n")
# 2. Check the distribution of Linear Regression predictions
plt.figure(figsize=(8, 5))
plt.hist(y_pred_linreg, bins=30, color='red', alpha=0.7)
plt.title(f'Figure {figure_number}: Distribution of Linear Regression Predictions')
plt.xlabel('Predicted Values')
plt.ylabel('Frequency')
plt.savefig(f"Figure {figure_number} - Distribution LinReg Predictions.png")
figure_number += 1
# 3. Compare the predictions with the actual values (Diabetes target)
plt.figure(figsize=(8, 5))
plt.scatter(np.arange(len(y_test)), y_test, color='black', label='Actual Values', alpha=0.6)
plt.scatter(np.arange(len(y_test)), y_pred_linreg, color='red', label='Linear Regression Predictions', alpha=0.6)
plt.title(f'Figure {figure_number}: Actual Values vs Linear Regression Predictions based on stepwise model selection')
plt.xlabel('Sample Index')
plt.ylabel('Diabetes (Actual) vs Predicted Value (Linear Regression)')
plt.legend()
plt.savefig(f"Figure {figure_number} - Actual Values vs Linear Regression Predictions based on stepwise model selection.png")
figure_number += 1
plt.figure(figsize=(12, 8))
# Taking a closer at the frequency distribution of the predicted values for linear regression, we can see that the range of values for 
# linear regression exceeds the range of 0 to 1. This makes logistics regression a better method in this regard.


# Section 6. Random chance baseline for Diabetes classification
# Provides a naive benchmark to compare against the trained models.
np.random.seed(42)
states = np.array([0, 1])
N = len(y_test)
random_predictions = np.random.choice(states, size=N, p=[0.5, 0.5])
random_accuracy = accuracy_score(y_test, random_predictions)
random_conf_matrix = confusion_matrix(y_test, random_predictions)
print("===== Random chance model performance =====")
print("Random chance model accuracy:", random_accuracy)
print("Random chance confusion matrix:\n", random_conf_matrix)
print(f"=========================================================\n")