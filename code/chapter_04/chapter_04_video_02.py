#####################
### Tuning models ###
#####################

import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Load the training data
X_train, X_test, y_train, y_test, feature_names = joblib.load("data/model_data.joblib")

# Define the parameter grid for XGBoost
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(
    eval_metric='logloss', random_state=42
)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='accuracy', cv=3, verbose=1)

grid_search.fit(X_train, y_train)   

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

joblib.dump(grid_search.best_estimator_, "models/xgb_model_tuned.joblib")