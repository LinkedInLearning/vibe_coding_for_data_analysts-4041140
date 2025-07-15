#############################
### Converting data types ###
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
