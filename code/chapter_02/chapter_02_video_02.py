#############################
### Converting data types ###
#############################

import pandas as pd

songs_joined = pd.read_csv('data/songs_joined.csv')
# Remove the time from release_week and convert to date
if 'release_week' in songs_joined.columns:
    songs_joined['release_week'] = pd.to_datetime(songs_joined['release_week']).dt.date


songs_joined['release_week']
    
# Convert genre to a categorical variable
if 'Genre' in songs_joined.columns:
    songs_joined['Genre'] = songs_joined['Genre'].astype('category')
