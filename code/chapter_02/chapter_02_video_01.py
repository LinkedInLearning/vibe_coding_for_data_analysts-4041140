##############################
###        Generating      ###
### descriptive statistics ###
##############################

import polars as pl

pl.Config.set_tbl_cols(20)

songs = pl.read_csv("data/songs.csv")

stats = songs.describe()

stats
