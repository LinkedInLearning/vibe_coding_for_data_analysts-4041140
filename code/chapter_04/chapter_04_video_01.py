#########################
### Testing different ###
###  baseline models  ###
#########################

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load data
X_train = joblib.load("data/songs_scaled_train.joblib")
X_test = joblib.load("data/songs_scaled_test.joblib")
y_train = joblib.load("data/number_1_hit_train.joblib")
y_test = joblib.load("data/number_1_hit_test.joblib")

# Linear Model
linear_model = LogisticRegression(max_iter=1000)
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
print("Linear Model Accuracy:", accuracy_score(y_test, y_pred_linear))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))