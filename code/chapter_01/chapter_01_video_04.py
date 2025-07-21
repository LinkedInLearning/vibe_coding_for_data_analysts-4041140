######################
### Generating SQL ###
######################

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

# Read all columns from the songs table in the SQLite database into a DataFrame
import sqlite3
conn = sqlite3.connect('data/songs_database.db')
songs_df = pd.read_sql_query('SELECT * FROM songs', conn)
conn.close()

songs_df = songs_df[songs_df['highest_rank'] < 25]

# Use duckdb to select all columns from the songs table and filter rows
import duckdb
duckdb_df = duckdb.query("SELECT * FROM 'data/songs_database.db'.songs WHERE highest_rank < 25").to_df()

