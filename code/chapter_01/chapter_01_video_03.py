##########################
### Loading data from  ###
###    html tables     ###
##########################

import pandas as pd
import polars as pl

song_lyric_features = pl.read_csv(
    "data/song_lyric_features.csv"
)

link = 'https://4041140.youcanlearnit.net/'

song_genre = pd.read_html(link)

song_genre = song_genre[0]

