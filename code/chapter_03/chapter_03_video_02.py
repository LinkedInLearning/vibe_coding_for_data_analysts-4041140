######################################
### Encoding categorical variables ###
######################################

import polars as pl
import sklearn.preprocessing as skp
from sklearn.impute import SimpleImputer

songs = pl.read_csv("data/songs_complete.csv")

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
    songs.select(pl.col(pl.Float64, pl.Int64)).columns +
    encoder.get_feature_names_out().tolist()
)

features_df = pl.DataFrame(X, schema=feature_names)