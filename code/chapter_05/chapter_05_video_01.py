############################
### Creating ridge plots ###
############################

import pandas as pd
import matplotlib.pyplot as plt
import joypy

# Read the CSV file
df = pd.read_csv("data/songs_joined.csv")

# Ensure 'Genre' and 'sentiment' columns exist
if 'Genre' in df.columns and 'sentiment' in df.columns:
    plt.figure(figsize=(12, 8))
    joypy.joyplot(
        df,
        by="Genre",
        column="sentiment",
        figsize=(12, 8),
        legend=False,
        colormap=plt.cm.Set2
    )
    plt.title("Ridgeline Chart of Sentiment by Genre")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.show()
else:
    print("Required columns 'Genre' and 'sentiment' not found in the data.")
