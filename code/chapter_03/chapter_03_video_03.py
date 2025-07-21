#########################
### Partitioning data ###
#########################

import pandas as pd

songs_joined = pd.read_csv('data/songs_joined.csv')
# Remove the time from release_week and convert to date
if 'release_week' in songs_joined.columns:
    songs_joined['release_week'] = pd.to_datetime(songs_joined['release_week']).dt.date


songs_joined['release_week']
    
# Convert genre to a categorical variable
if 'Genre' in songs_joined.columns:
    songs_joined['Genre'] = songs_joined['Genre'].astype('category')

# Show correlations between numeric variables
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

# Perform feature scaling on selected columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scale_cols = [
    'sentiment', 'word_count', 'profanity_count', 'producer_count',
    'songwriter_count', 'smog_index', 'difficult_words', 'proportion_profanity'
]
available_scale_cols = [col for col in scale_cols if col in songs_joined.columns]
if available_scale_cols:
    songs_joined[[f'{col}_scaled' for col in available_scale_cols]] = scaler.fit_transform(songs_joined[available_scale_cols])

# Create a new DataFrame with only the scaled features
if available_scale_cols:
    songs_scaled = pd.DataFrame(
        scaler.fit_transform(songs_joined[available_scale_cols]),
        columns=[f'{col}_scaled' for col in available_scale_cols],
        index=songs_joined.index
    )
    print('\nScaled features DataFrame:')
    print(songs_scaled.head())

# Impute missing data in songs_scaled using sklearn's IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

if 'songs_scaled' in locals() and songs_scaled is not None:
    imputer = IterativeImputer(random_state=0)
    songs_scaled[:] = imputer.fit_transform(songs_scaled)
    print('\nImputed songs_scaled DataFrame (IterativeImputer):')
    print(songs_scaled.head())

# Categorical feature encoding for the genre variable
if 'Genre' in songs_joined.columns:
    genre_encoded = pd.get_dummies(songs_joined['Genre'], prefix='genre')
    songs_joined = pd.concat([songs_joined, genre_encoded], axis=1)
    print('\nEncoded genre columns:')
    print(genre_encoded.head())

# Split the scaled features and number_1_hit target into training and testing sets and dump with joblib
from sklearn.model_selection import train_test_split
import joblib

if 'songs_scaled' in locals() and songs_scaled is not None and 'number_1_hit' in songs_joined.columns:
    X = songs_scaled
    y = songs_joined['number_1_hit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    joblib.dump(X_train, 'data/songs_scaled_train.joblib')
    joblib.dump(X_test, 'data/songs_scaled_test.joblib')
    joblib.dump(y_train, 'data/number_1_hit_train.joblib')
    joblib.dump(y_test, 'data/number_1_hit_test.joblib')
    print('\nTraining and testing sets (features and target) have been saved as joblib files.')

