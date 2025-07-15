############################
### Creating ridge plots ###
############################

import polars as pl
from ridgeplot import ridgeplot

songs = pl.read_csv("data/songs_complete.csv")

plot_data = songs.select([
    pl.col("sentiment"), 
    pl.col("number_1_hit")
])

samples = []

for value in songs["number_1_hit"].unique().to_list():
    samples.append(songs.filter(pl.col("number_1_hit") == value).select("sentiment").drop_nulls().to_numpy().flatten())

fig = ridgeplot(samples=samples)

fig.show()
