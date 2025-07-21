##########################
### Loading data from  ###
###    html tables     ###
##########################

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

link = 'https://4041140.youcanlearnit.net/'

# Read the HTML table from the link using pandas
html_tables = pd.read_html(link)
# If there are multiple tables, the first one can be accessed as html_tables[0]
first_table = html_tables[0]