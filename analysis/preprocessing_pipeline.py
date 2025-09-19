"""
preprocessing_pipeline.py

Reads the raw housing training set (13 columns).
Applies a preprocessing pipeline using scikit-learn (imputation, scaling, encoding, feature engineering).
Saves the final processed dataset (24 features) to the training folder.

"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load raw training data
train_set = pd.read_csv("data/train/housing_train.csv")

housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# Feature Engineering
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# Define numerical & categorical columns
num_attribs = list(housing.drop("ocean_proximity", axis=1))
cat_attribs = ["ocean_proximity"]

# Create preprocessing pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
])

# Apply preprocessing pipeline
housing_prepared = full_pipeline.fit_transform(housing)

# Get feature names
num_features = num_attribs
cat_features = list(full_pipeline.named_transformers_["cat"].get_feature_names_out(cat_attribs))
feature_names = num_features + cat_features

# Create DataFrame
housing_processed = pd.DataFrame(housing_prepared, columns=feature_names, index=housing.index)

# Add target back
housing_processed["median_house_value"] = housing_labels

# Save processed dataset
output_path = "data/train/housing_train_processed.csv"
housing_processed.to_csv(output_path, index=False)

