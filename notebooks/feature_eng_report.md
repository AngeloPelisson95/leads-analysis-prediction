# Feature Engineering Module Report

## File: `src/features/feature_engineering.py`

### Overview
This module provides utilities for feature engineering in a machine learning pipeline, focusing on:
- Flag clustering (using Jenks Natural Breaks)
- Feature scaling
- Categorical encoding (label and target encoding)
- Handling missing values
- Outlier removal
- Data cleaning and preparation

---

## Main Classes & Functions

### 1. `FeatureEngineering`
- **Purpose:** General feature engineering pipeline.
- **Key Methods:**
  - `create_flag_clustering_features()`: Groups data by flag variable combinations, computes statistics, and creates a `sum_flags` feature.
- **Usage:** Used to generate new features based on flag columns and their combinations.

### 2. `FlagClusteringTransformer`
- **Purpose:** Production-ready transformer for flag clustering features, compatible with scikit-learn pipelines.
- **Key Features:**
  - Uses Jenks Natural Breaks to cluster flag combinations by their mean target value.
  - Can be fitted on training data and used to transform new data without access to historical data.
  - Handles cases where Jenks clustering is not possible (fallback to simple binning).
- **Key Methods:**
  - `fit()`: Learns clusters from training data.
  - `transform()`: Applies learned clusters to new data.
  - `fit_transform()`: Convenience method for fitting and transforming in one step.

### 3. `PreprocessingFeatures`
- **Purpose:** Comprehensive transformer for feature engineering, including:
  - Missing value handling
  - Numerical feature scaling
  - Categorical encoding (label and target encoding)
  - Flag clustering features
  - Outlier removal
  - Data cleaning (duplicate/spurious value removal)
- **Key Methods:**
  - `_location_split()`: Splits a combined location column into `city` and `state`.
  - `_flag_to_int()`: Converts flag columns to integer type.
  - `_fuel_type_to_flag()`: Converts a fuel type column into multiple flag columns.
  - `_remove_duplicated()`: Removes duplicate columns and rows.
  - `_clean_spurious_values()`: Removes rows with spurious values (e.g., -1).
  - `remove_outliers_iqr()`: Removes outliers using a percentile-based approach.
  - `_feature_setup()`: Sets up features for the transformer.

---

## Notable Implementation Details
- **Logging:** Uses Python's `logging` module for detailed process tracking and warnings.
- **Serialization:** Transformers are designed to be serializable (e.g., with pickle) for production use.
- **Sklearn Compatibility:** `FlagClusteringTransformer` inherits from `BaseEstimator` and `TransformerMixin` for seamless integration with scikit-learn pipelines.
- **Robustness:** Handles edge cases such as missing values, unseen flag combinations, and duplicate columns.


## Summary
- The module provides a robust, production-ready set of tools for feature engineering, especially for datasets with many binary flag features.
- It supports both exploratory data analysis and deployment in machine learning pipelines.
- The design emphasizes modularity, reusability, and compatibility with the scikit-learn ecosystem.

---

