####################
### Merging Data ###
####################

import sqlite3
import pandas as pd
import polars as pl

song_lyric_features = pl.read_csv(
    "data/song_lyric_features.csv"
)

link = 'https://4041140.youcanlearnit.net/'

song_genre = pd.read_html(link)

song_genre = song_genre[0]

conn = sqlite3.connect(
            database="data/songs_database.db",
)

cursor = conn.cursor()

cursor.execute(f"PRAGMA table_info(songs)")

column_info = cursor.fetchall()

col_names = [column[1] for column in column_info]

cursor.execute("SELECT * FROM songs")

song_rows = cursor.fetchall()

song_weeks = pd.DataFrame(song_rows, columns=col_names)

cursor.close()

conn.close()

song_personnel = pd.read_feather("data/song_personnel.feather")

songs = pd.merge(
    song_lyric_features.to_pandas(), song_genre, 
    how='left', left_on='song_id', right_on='Song ID'
)

songs = songs.merge(
    song_weeks, on='song_id'
)

songs = songs.merge(
    song_personnel, on='song_id'
)

songs.to_csv("data/songs.csv", index=False)

