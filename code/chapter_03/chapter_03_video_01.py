############################
### Scaling and imputing ###
###   data for sklearn   ###
############################

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

