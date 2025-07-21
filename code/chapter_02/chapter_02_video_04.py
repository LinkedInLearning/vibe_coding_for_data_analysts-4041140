########################
### Aggregating data ###
########################

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
'''
AUTHOR NOTE:
The following code while technically correct, will not include
the necessary columns names and needed prompted again to produce
the correct output.
'''
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
'''
AUTHOR NOTE:
The following code is the result of the second prompt
and will run as intended and expected.
'''
selected_stats = ['sentiment', 'word_count', 'producer_count', 'songwriter_count']
available_stats = [col for col in selected_stats if col in songs_joined.columns]
if 'Genre' in songs_joined.columns and available_stats:
    means_stats = songs_joined.groupby('Genre')[available_stats].mean()
    stds_stats = songs_joined.groupby('Genre')[available_stats].std()
    print('\nMeans for sentiment, word_count, producer_count, songwriter_count by Genre:')
    print(means_stats)
    print('\nStandard Deviations for sentiment, word_count, producer_count, songwriter_count by Genre:')
    print(stds_stats)
