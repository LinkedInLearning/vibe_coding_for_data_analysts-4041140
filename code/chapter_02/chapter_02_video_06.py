##################################
### Creating derived variables ###
##################################

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

songs.select(numeric_cols).to_pandas().corr()


# Find the means and standard deviations for the numeric variables for each group in the genre variable.

means = songs.group_by("Genre").agg(
    [pl.col(col).mean().alias(f"{col}_mean") for col in numeric_cols.columns]
)

stddevs = songs.group_by("Genre").agg(
    [pl.col(col).std().alias(f"{col}_stddev") for col in numeric_cols.columns]
)

means.join(stddevs, on="Genre")


# Melt the df data so that song_id is the index and the 'sentiment', 'word_count', 'profanity_count', and 'smog_index' variables are everything else.

melted = songs.unpivot(
    on=["sentiment", "word_count", "profanity_count", "smog_index"],
    index="song_id"
)   

songs = songs.with_columns(
    (pl.col("weeks_on_charts") / 52).cast(pl.Float64).alias("years_on_charts")
)

songs = songs.with_columns(
    (pl.col("profanity_count") / pl.col("word_count")).cast(pl.Float64).alias("profanity_proportion")
)

songs = songs.with_columns(
    (pl.col("highest_rank") == 1).cast(pl.Boolean).alias("number_1_hit")
)

songs.write_csv("data/songs_complete.csv")
songs.write_parquet("data/songs_complete.parquet")
