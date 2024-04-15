# Import necessary libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
import numpy as np

# Fetching the dataset from the UCI Machine Learning Repository
wine_quality = fetch_ucirepo(id=186)

# Extracting features and target from the dataset
X = wine_quality.data.features
y = wine_quality.data.targets

# Check if y is a DataFrame and convert it to a numpy array if necessary
if isinstance(y, pd.DataFrame):
    y = y.squeeze()  # This converts a single-column DataFrame to a Series

# Reshape y to be a 1D array
y = y.values.ravel()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
selector = SelectKBest(f_classif, k=10)  # Selecting the top 10 features
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Model Selection and Hyperparameter Tuning
svm_model = SVC(random_state=42)
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
# Ensuring at least 2 splits for cross-validation
n_splits = max(2, np.min(np.bincount(y_train)))

grid_search = GridSearchCV(svm_model, param_grid, cv=n_splits, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

# Best model parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Cross-validation of the best model
cross_val_scores = cross_val_score(grid_search.best_estimator_, X_train_selected, y_train, cv=n_splits)

# Output best parameters and cross-validation results
print("Best Parameters:", best_params)
print("Best Cross-validation Score:", best_score)
print("Cross-validation Scores:", cross_val_scores)
