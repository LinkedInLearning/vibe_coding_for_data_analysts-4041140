##################################
### Creating derived variables ###
##################################

import pandas as pd

songs_joined = pd.read_csv('data/songs_joined.csv')
# Remove the time from release_week and convert to date
if 'release_week' in songs_joined.columns:
    songs_joined['release_week'] = pd.to_datetime(songs_joined['release_week']).dt.date


songs_joined['release_week']
    
# Convert genre to a categorical variable
if 'Genre' in songs_joined.columns:
    songs_joined['Genre'] = songs_joined['Genre'].astype('category')

print(songs_joined.corr(numeric_only=True))

# Visualize the correlations using a corrplot
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = songs_joined.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.show()

# Find means and standard deviations for lyric features and personnel variables by genre
lyric_features = [col for col in songs_joined.columns if 'lyric' in col.lower()]
personnel_features = [col for col in songs_joined.columns if 'personnel' in col.lower()]
selected_features = lyric_features + personnel_features

# If 'Genre' exists, group by it and calculate stats
if 'Genre' in songs_joined.columns and selected_features:
    means = songs_joined.groupby('Genre')[selected_features].mean()
    stds = songs_joined.groupby('Genre')[selected_features].std()
    print('Means by Genre:')
    print(means)
    print('\nStandard Deviations by Genre:')
    print(stds)

# Means and standard deviations for selected features by genre
selected_stats = ['sentiment', 'word_count', 'producer_count', 'songwriter_count']
available_stats = [col for col in selected_stats if col in songs_joined.columns]
if 'Genre' in songs_joined.columns and available_stats:
    means_stats = songs_joined.groupby('Genre')[available_stats].mean()
    stds_stats = songs_joined.groupby('Genre')[available_stats].std()
    print('\nMeans for sentiment, word_count, producer_count, songwriter_count by Genre:')
    print(means_stats)
    print('\nStandard Deviations for sentiment, word_count, producer_count, songwriter_count by Genre:')
    print(stds_stats)

# Melt the songs data
melt_vars = ['sentiment', 'word_count', 'profanity_count', 'smog_index']
id_var = 'song_id'
available_melt_vars = [col for col in melt_vars if col in songs_joined.columns]
if id_var in songs_joined.columns and available_melt_vars:
    songs_melted = pd.melt(
        songs_joined,
        id_vars=[id_var],
        value_vars=available_melt_vars,
        var_name='feature',
        value_name='value'
    )
    songs_melted.set_index(id_var, inplace=True)
    print('\nMelted songs data:')
    print(songs_melted.head())

# Create a new variable called years_on_charrrrrrrrrrts from weeks_on_charts
if 'weeks_on_charts' in songs_joined.columns:
    songs_joined['years_on_charts'] = songs_joined['weeks_on_charts'] / 52

# Add a new variable for the proportion of words that are profanity
if 'profanity_count' in songs_joined.columns and 'word_count' in songs_joined.columns:
    songs_joined['proportion_profanity'] = songs_joined['profanity_count'] / songs_joined['word_count']
# Add an indicator variable for number 1 hit
if 'highest_rank' in songs_joined.columns:
    songs_joined['number_1_hit'] = (songs_joined['highest_rank'] == 1).astype(int)
