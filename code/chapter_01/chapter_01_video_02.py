################################
### Loading data with polars ###
################################

import pandas as pd

df = pd.read_csv(
    'data/song_lyric_features.csv',
    engine='pyarrow',
    dtype_backend='pyarrow'
)

# Read the CSV file into a Polars DataFrame
import polars as pl
pl_df = pl.read_csv('data/song_lyric_features.csv')

# calculating the mean sentiment

mean_sentiment = pl_df.select(
    pl.col("sentiment").mean()
    ).to_numpy()[0][0]
