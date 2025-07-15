##########################
### Loading data with  ###
### pandas and pyarrow ###
##########################

import pandas as pd
import pyarrow as pa

song_lyric_features = pd.read_csv(
    "data/song_lyric_features.csv", 
    dtype_backend='pyarrow', 
    engine='pyarrow'
)
