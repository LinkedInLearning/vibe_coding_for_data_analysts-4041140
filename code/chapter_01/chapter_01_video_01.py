##########################
### Loading data with  ###
### pandas and pyarrow ###
##########################

import pandas as pd
import pyarrow as pa

df = pd.read_csv(
    'data/song_lyric_features.csv',
    engine='pyarrow',
    dtype_backend='pyarrow'
)