############################
###  Creating a neural   ###
### network with pytorch ###
############################

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

# Ready to use with PyTorch Tabular models
from pytorch_tabular.config import ModelConfig, OptimizerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.tabular_model import TabularModel
import pandas as pd

# Example: Train a TabNet model with PyTorch Tabular

# Prepare training data for PyTorch Tabular
train_df = X_train.copy()
train_df['number_1_hit'] = y_train.values

# ModelConfig (using TabNet as example)
model_config = CategoryEmbeddingModelConfig(
    task="classification",
    learning_rate=1e-3,
    layers="128-64-32",
    activation="LeakyReLU"
    # Add other model-specific parameters as needed
)

# OptimizerConfig
optimizer_config = OptimizerConfig()

# Initialize and train the model
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train_df)

# Evaluate model performance on the test set
'''
AUTHOR NOTE: The code below is the result of a second attempt
to prompt our vibe coding assistant to generate accuracy
scores for our pytorch tabular model. Just like the first
attempt, it failed to generate correct code.
'''
test_df = X_test.copy()
test_df['number_1_hit'] = y_test.values
results = tabular_model.evaluate(test=test_df)
print("\nModel Performance on Test Set:")
print(results)

'''
AUTHOR NOTE: The code below is a proper example of
how to obtain accuracy scores for a pytorch tabular model.
Like so many vibe coding tasks, this one needed
a human to regain control and write some code.
'''

# Proper way to calculate accuracy for PyTorch Tabular model
from sklearn.metrics import accuracy_score, classification_report

# Prepare test data for predictions (without target column for predictions)
test_data_for_prediction = X_test.copy()

# Make predictions on the test set
predictions = tabular_model.predict(test_data_for_prediction)
y_pred = predictions['number_1_hit_prediction'].values

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

# Optional: Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

