################################
### Creating explainer plots ###
################################

import joblib
import polars as pl
import pandas as pd
import shap

X_train, X_test, y_train, y_test, feature_names = joblib.load("data/model_data.joblib")

X_train = pd.DataFrame(X_train)

xgb_model = joblib.load("models/xgb_model.joblib")

explainer = shap.Explainer(xgb_model.predict, shap.utils.sample(X_train, 1000))
shap_values = explainer(X_train.sample(10))

shap.plots.waterfall(shap_values[0])
