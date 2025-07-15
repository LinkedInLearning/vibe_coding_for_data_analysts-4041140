#########################
### Partitioning data ###
#########################

import pandas as pd
import polars as pl
import sklearn.preprocessing as skp
from sklearn.impute import SimpleImputer

songs = pl.read_parquet("data/songs_complete.parquet")

selected_features = [
    "sentiment",
    "word_count",
    "profanity_count",
    "producer_count",
    "songwriter_count",
    "smog_index",
    "difficult_words",
    "profanity_proportion"
]

scaler = skp.StandardScaler()

scaled_features = scaler.fit_transform(
    songs.select(selected_features).to_numpy()
)

# Impute missing values
imputer = SimpleImputer(strategy="mean")

imputed_features = imputer.fit_transform(scaled_features)

# Encode categorical variables
encoder = skp.OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_features = encoder.fit_transform(
    songs.select(pl.col(pl.Categorical)).to_numpy()
)

import numpy as np

X = np.hstack([imputed_features, encoded_features]) 

feature_names = (
    selected_features +
    encoder.get_feature_names_out().tolist()
)

import re

feature_names = [re.sub(r"^x0_", "", col) for col in feature_names]

features_df = pd.DataFrame(X, columns=feature_names)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features_df, songs.select("number_1_hit").to_numpy().ravel(), 
    test_size=0.2, random_state=42
)

import joblib

joblib.dump([X_train, X_test, y_train, y_test, feature_names], "data/model_data.joblib")
