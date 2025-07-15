################################
### Loading data with polars ###
################################

import polars as pl

song_lyric_features = pl.read_csv(
    "data/song_lyric_features.csv"
)

# calculating the mean sentiment

mean_sentiment = song_lyric_features.select(
    pl.col("sentiment").mean()
    ).to_numpy()[0][0]