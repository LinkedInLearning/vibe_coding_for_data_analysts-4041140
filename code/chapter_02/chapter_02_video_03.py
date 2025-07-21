#############################
### Examning correlations ###
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

