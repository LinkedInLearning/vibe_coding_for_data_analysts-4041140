###############################
###    Creating a tabular   ###
### data loader for pytorch ###
###############################

import polars as pl
# pip install "pytorch_tabular[extra]"
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

# Load the complete data
complete_data = pl.read_parquet("data/songs_complete.parquet")

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
    categorical_cols=["genre"], 
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
