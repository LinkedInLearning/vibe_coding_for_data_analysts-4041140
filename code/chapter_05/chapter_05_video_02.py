################################
### Creating explainer plots ###
################################

import joblib
import shap
import pandas as pd

# Load the trained XGBoost model
xgb_model = joblib.load("data/xgb_model.joblib")

# Load test data
X_test = joblib.load("data/songs_scaled_test.joblib")

# Create SHAP explainer and values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Detailed beeswarm plot
shap.summary_plot(shap_values, X_test)
