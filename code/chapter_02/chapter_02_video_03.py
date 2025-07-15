#############################
### Examning correlations ###
#############################

import polars as pl

pl.Config.set_tbl_cols(20)

songs = pl.read_csv("data/songs.csv")

songs = songs.with_columns(
    pl.col("release_week").str.replace(" 00:00:00", "")
)

songs = songs.with_columns(
    pl.col("release_week").str.strptime(pl.Date, "%Y-%m-%d")
)

songs = songs.with_columns(
    pl.col("Genre").cast(pl.Categorical)
)

numeric_cols = songs.select(pl.col(pl.Float64, pl.Int64))

correlations = songs.select(numeric_cols).to_pandas().corr()

# visualize the correlation matrix

import seaborn as sns
import matplotlib.pyplot as plt 

plt.figure(figsize=(10, 8))

sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", square=True)

plt.show()
