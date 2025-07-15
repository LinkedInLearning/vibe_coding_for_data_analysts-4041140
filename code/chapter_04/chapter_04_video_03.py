############################
###  Creating a neural   ###
### network with pytorch ###
############################

import joblib
import numpy as np
import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

# Load the complete data
X_train, X_test, y_train, y_test, feature_names = joblib.load("data/model_data.joblib")

training = np.hstack((X_train, y_train.reshape(-1, 1)))
training = pd.DataFrame(training, columns=feature_names + ["number_1_hit"])

testing = np.hstack((X_test, y_test.reshape(-1, 1)))
testing = pd.DataFrame(testing, columns=feature_names + ["number_1_hit"])
testing.columns
# Define the data configuration
data_config = DataConfig(
    target=["number_1_hit"],
    continuous_cols=[
        "sentiment",
        "word_count",
        "profanity_count",
        "producer_count",
        "songwriter_count",
        "smog_index",
        "difficult_words",
        "profanity_proportion",
    ],
    categorical_cols=['Alternative', 'Christian', 'Country',
       'Hip-Hop', 'Pop', 'Rock'], 
    normalize_continuous_features=True
)

trainer_config = TrainerConfig(
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
)
optimizer_config = OptimizerConfig()

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="1024-512-512",
    activation="LeakyReLU",
    learning_rate=1e-3,
)

# Create the model
model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

# Fit the model
model.fit(train=training, validation=testing)
