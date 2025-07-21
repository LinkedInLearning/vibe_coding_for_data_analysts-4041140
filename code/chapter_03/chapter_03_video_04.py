###############################
###    Creating a tabular   ###
### data loader for pytorch ###
###############################

import joblib
from pytorch_tabular.config import DataConfig, TrainerConfig

# Load training and testing data
X_train = joblib.load("data/songs_scaled_train.joblib")
X_test = joblib.load("data/songs_scaled_test.joblib")
y_train = joblib.load("data/number_1_hit_train.joblib")
y_test = joblib.load("data/number_1_hit_test.joblib")

# Prepare DataConfig
data_config = DataConfig(
    target=['number_1_hit'],
    continuous_cols=list(X_train.columns),
    categorical_cols=[],
)

# Prepare TrainerConfig
trainer_config = TrainerConfig(
    auto_lr_find=True,
    batch_size=64,
    max_epochs=20,
    accelerator="cpu",  # Use "gpu" if available
)