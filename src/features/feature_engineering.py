"""
Feature engineering utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from jenkspy import JenksNaturalBreaks
import pickle
import os
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Feature engineering pipeline."""

    def __init__(self, data: pd.DataFrame, target_col: str, flag_cols: list = []):
        self.scaler = StandardScaler()
        self.target_col = target_col
        self.flag_cols = flag_cols
        self.feature_selector = None
        self.is_scaler_fitted = False
        self.fitted_numeric_columns = None
        self.data = data

    
    def create_flag_clustering_features(
        self
    ) -> pd.DataFrame:
        """
        Create clustering features based on flag variables combinations.

        This function groups data by unique combinations of flag variables,
        calculates statistics, and creates a sum_flags feature that represents
        the total number of active flags for each record.

        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target variable column name

        Returns:
            pd.DataFrame: Data with additional flag clustering features
        """
        data_enhanced = self.data.copy()
        # Identify flag columns
        if len(self.flag_cols) == 0:
            logger.warning("No flag columns found in the dataset")
            raise ValueError("No flag columns passed for argument flag_cols")

        # Remove target flag if present
        feature_flag_cols = [col for col in self.flag_cols if col != "flg_leads"]

        if len(feature_flag_cols) == 0:
            logger.warning("No feature flag columns found (excluding target)")
            return data_enhanced

        logger.info(
            f"Processing {len(feature_flag_cols)} flag columns: {feature_flag_cols}"
        )

        # Create sum of flags feature
        data_enhanced["sum_flags"] = data_enhanced[feature_flag_cols].sum(axis=1)

        grouped = (
            data.groupby(feature_flag_cols)
            .agg(
                n_ads=(self.target_col, "count"),
                mean_leads=(self.target_col, "mean"),
                sum_leads=(self.target_col, "sum"),
            )
            .reset_index()
        )

        # Sort by sum leads and calculate proportion
        grouped_sorted = grouped.sort_values(by="sum_leads", ascending=False)
        total_leads = self.data[self.target_col].sum()
        grouped_sorted["proportion"] = (grouped_sorted["sum_leads"] / total_leads) * 100

        # Create sum_flags for grouped data
        flg_cols = [col for col in grouped_sorted.columns if col.startswith("flg_")]

        grouped_sorted["sum_flags"] = grouped_sorted[flg_cols].sum(axis=1)

        # Calculate final group statistics by sum_flags
        self.final_group = (
            grouped_sorted.groupby("sum_flags")
            .agg({"n_ads": "sum", "mean_leads": "mean", "sum_leads": "sum"})
            .reset_index()
            .sort_values(by="sum_flags", ascending=False)
        )
        logger.info(
            f"Grouped data by flag combinations, resulting in {len(self.final_group)} unique sum_flags"
        )
        logger.info("Flag clustering features created successfully")
        return self.final_group

class FlagClusteringTransformer(BaseEstimator, TransformerMixin):
    """
    Production-ready transformer for flag clustering features.

    This transformer can be fitted on training data and then used to transform
    new data in production without needing access to historical data.
    """

    def __init__(self, n_clusters=5, target_col="leads", feature_flag_cols=[]):
        """
        Initialize the transformer.

        Args:
            n_clusters (int): Number of clusters for Jenks Natural Breaks
            target_col (str): Target column name for training
        """
        self.n_clusters = n_clusters
        self.target_col = target_col
        self.feature_flag_cols = feature_flag_cols
        self.jenks_model = None
        self.jenks_col = "mean_leads"
        self.cluster_mapping = None
        self.breaks_ = None
        self.is_fitted = False
        self.feature_engineering = FeatureEngineering()

    def fit(self, X, y=None):
        """
        Fit the transformer on training data.

        Args:
            X (pd.DataFrame): Training data
            y: Not used, present for sklearn compatibility

        Returns:
            self: Returns self for method chaining
        """
        logger.info("Fitting FlagClusteringTransformer...")
        logger.info(f"Using flag columns: {self.feature_flag_cols}")

        self.final_group = self.feature_engineering.create_flag_clustering_features(
            X, target_col=self.target_col, flag_cols=self.feature_flag_cols
        )
        # Fit Jenks Natural Breaks if we have enough data points
        if len(self.final_group) > 1:
            try:
                actual_clusters = min(self.n_clusters, len(self.final_group))
                self.jenks_model = JenksNaturalBreaks(actual_clusters)
                x = self.final_group[self.jenks_col].values
                self.jenks_model.fit(x)
                self.breaks_ = self.jenks_model.breaks_

                # Create cluster mapping based on sum_flags ranges
                self.cluster_mapping = {}
                for i, label in enumerate(self.jenks_model.labels_):
                    sum_flags_value = self.final_group.iloc[i]["sum_flags"]
                    self.cluster_mapping[sum_flags_value] = label

                logger.info(f"Fitted Jenks model with {actual_clusters} clusters")
                logger.info(f"Cluster breaks: {self.breaks_}")

            except Exception as e:
                logger.warning(f"Failed to fit Jenks model: {e}. Using simple binning.")
                self._create_simple_binning(self.final_group["sum_flags"])
        else:
            logger.warning(
                "Not enough data points for clustering. Using simple binning."
            )
            self._create_simple_binning(self.final_group["sum_flags"])

        self.is_fitted = True
        logger.info("FlagClusteringTransformer fitted successfully")
        return self

    def _create_simple_binning(self, sum_flags_series):
        """Create simple binning as fallback when Jenks fails."""
        min_val = sum_flags_series.min()
        max_val = sum_flags_series.max()

        if min_val == max_val:
            # All values are the same
            self.cluster_mapping = {min_val: 0}
            self.breaks_ = [min_val, max_val]
        else:
            # Create equal-width bins
            bins = np.linspace(min_val, max_val, self.n_clusters + 1)
            self.breaks_ = bins.tolist()

            # Create mapping for each possible sum_flags value
            unique_vals = sum_flags_series.unique()
            self.cluster_mapping = {}
            for val in unique_vals:
                cluster_id = np.digitize(val, bins[1:-1])  # Exclude first and last
                self.cluster_mapping[val] = min(cluster_id, self.n_clusters - 1)

    def transform(self, X):
        """
        Transform new data using fitted parameters.

        Args:
            X (pd.DataFrame): Data to transform

        Returns:
            pd.DataFrame: Transformed data with flag clustering features
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()

        if self.feature_flag_cols is None or len(self.feature_flag_cols) == 0:
            logger.warning("No flag columns to process")
            return X_transformed

        # Create sum_flags feature
        available_flag_cols = [
            col for col in self.feature_flag_cols if col in X.columns
        ]
        if len(available_flag_cols) == 0:
            logger.warning("None of the fitted flag columns found in new data")
            return X_transformed

        X_transformed["sum_flags"] = X_transformed[available_flag_cols].sum(axis=1)

        # Apply cluster mapping
        if self.cluster_mapping:
            X_transformed["flag_cluster"] = (
                X_transformed["sum_flags"].map(self.cluster_mapping).fillna(0)
            )  # Default to cluster 0 for unseen combinations
        else:
            X_transformed["flag_cluster"] = 0

        logger.info(f"Transformed data with {len(available_flag_cols)} flag columns")
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit the transformer and transform the data in one step.

        Args:
            X (pd.DataFrame): Training data
            y: Not used, present for sklearn compatibility

        Returns:
            pd.DataFrame: Transformed training data
        """
        return self.fit(X, y).transform(X)

    def save(self, filepath):
        """
        Save the fitted transformer to disk.
        
        Args:
            filepath (str): Path to save the transformer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted transformer")
        
        transformer_data = {
            'feature_flag_cols': self.feature_flag_cols,
            'cluster_mapping': self.cluster_mapping,
            'breaks_': self.breaks_,
            'n_clusters': self.n_clusters,
            'target_col': self.target_col,
            'is_fitted': self.is_fitted
        }
        
        # Save jenks model separately if it exists
        if self.jenks_model is not None:
            transformer_data['jenks_labels'] = self.jenks_model.labels_
            transformer_data['jenks_groups'] = self.jenks_model.groups_
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(transformer_data, f)
            
        logger.info(f"FlagClusteringTransformer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a fitted transformer from disk.
        
        Args:
            filepath (str): Path to load the transformer from
            
        Returns:
            FlagClusteringTransformer: Loaded transformer
        """
        with open(filepath, 'rb') as f:
            transformer_data = pickle.load(f)
        
        transformer = cls(
            n_clusters=transformer_data['n_clusters'],
            target_col=transformer_data['target_col'],
            feature_flag_cols=transformer_data['feature_flag_cols']
        )
        
        transformer.cluster_mapping = transformer_data['cluster_mapping']
        transformer.breaks_ = transformer_data['breaks_']
        transformer.is_fitted = transformer_data['is_fitted']
        
        # Reconstruct jenks model if data exists
        if 'jenks_labels' in transformer_data:
            transformer.jenks_model = JenksNaturalBreaks(transformer_data['n_clusters'])
            transformer.jenks_model.labels_ = transformer_data['jenks_labels']
            transformer.jenks_model.groups_ = transformer_data['jenks_groups']
            transformer.jenks_model.breaks_ = transformer_data['breaks_']
        
        logger.info(f"FlagClusteringTransformer loaded from {filepath}")
        return transformer


class ComprehensiveFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A comprehensive feature transformer that includes:
    - Missing value handling
    - Numerical feature scaling
    - Categorical encoding (label and target encoding)
    - Flag clustering features
    """
    
    def __init__(self, 
                 numeric_columns=None,
                 categorical_columns=None,
                 target_encode_columns=None,
                 flag_columns=None,
                 missing_strategy='mean',
                 n_flag_clusters=5,
                 target_col='leads'):
        """
        Initialize the comprehensive transformer.
        
        Args:
            numeric_columns (list): Columns to scale
            categorical_columns (list): Columns to label encode
            target_encode_columns (list): Columns to target encode
            flag_columns (list): Flag columns for clustering
            missing_strategy (str): Strategy for missing values
            n_flag_clusters (int): Number of clusters for flag clustering
            target_col (str): Target column name
        """
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target_encode_columns = target_encode_columns
        self.flag_columns = flag_columns
        self.missing_strategy = missing_strategy
        self.n_flag_clusters = n_flag_clusters
        self.target_col = target_col
        
        # Initialize components
        self.feature_engineering = FeatureEngineering()
        self.flag_transformer = None
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """
        Fit all transformers.
        
        Args:
            X (pd.DataFrame): Training data
            y (pd.Series): Target variable
            
        Returns:
            self: Returns self for method chaining
        """
        logger.info("Fitting ComprehensiveFeatureTransformer...")
        
        X_processed = X.copy()
        
        # 1. Handle missing values
        X_processed = self.feature_engineering.handle_missing_values(X_processed, self.missing_strategy)
        
        # 2. Target encoding (requires target variable)
        if self.target_encode_columns and y is not None:
            X_processed = self.feature_engineering.target_encode_categorical_variables(
                X_processed, y, self.target_encode_columns
            )
        
        # 3. Label encoding
        if self.categorical_columns:
            X_processed = self.feature_engineering.encode_categorical_variables(
                X_processed, self.categorical_columns
            )
        
        # 4. Scaling
        if self.numeric_columns:
            # Add target encoded columns to numeric columns if they exist
            extended_numeric = self.numeric_columns.copy() if self.numeric_columns else []
            if self.target_encode_columns:
                extended_numeric.extend([f'{col}_target_encoded' for col in self.target_encode_columns 
                                       if f'{col}_target_encoded' in X_processed.columns])
            
            X_processed = self.feature_engineering.scale_features(X_processed, extended_numeric, fit=True)
        
        # 5. Flag clustering
        if self.flag_columns:
            self.flag_transformer = FlagClusteringTransformer(
                n_clusters=self.n_flag_clusters,
                target_col=self.target_col,
                feature_flag_cols=self.flag_columns
            )
            X_processed = self.flag_transformer.fit_transform(X_processed)
        
        self.is_fitted = True
        logger.info("ComprehensiveFeatureTransformer fitted successfully")
        return self
    
    def transform(self, X):
        """
        Transform new data using fitted transformers.
        
        Args:
            X (pd.DataFrame): Data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        X_processed = X.copy()
        
        # 1. Handle missing values (using same strategy)
        X_processed = self.feature_engineering.handle_missing_values(X_processed, self.missing_strategy)
        
        # 2. Target encoding (using fitted encoders)
        if self.target_encode_columns:
            for col in self.target_encode_columns:
                if col in self.feature_engineering.target_encoders and col in X_processed.columns:
                    X_processed[f'{col}_target_encoded'] = self.feature_engineering.target_encoders[col].transform(
                        X_processed[[col]]
                    ).ravel()
        
        # 3. Label encoding (using fitted encoders)
        if self.categorical_columns:
            for col in self.categorical_columns:
                if col in self.feature_engineering.label_encoders and col in X_processed.columns:
                    # Handle unseen categories
                    try:
                        X_processed[col] = self.feature_engineering.label_encoders[col].transform(X_processed[col])
                    except ValueError:
                        # Handle unseen categories by setting them to a default value
                        logger.warning(f"Unseen categories found in {col}, setting to 0")
                        known_categories = set(self.feature_engineering.label_encoders[col].classes_)
                        X_processed[col] = X_processed[col].apply(
                            lambda x: x if x in known_categories else self.feature_engineering.label_encoders[col].classes_[0]
                        )
                        X_processed[col] = self.feature_engineering.label_encoders[col].transform(X_processed[col])
        
        # 4. Scaling (using fitted scaler)
        if self.numeric_columns and self.feature_engineering.is_scaler_fitted:
            # Include target encoded columns
            extended_numeric = self.numeric_columns.copy() if self.numeric_columns else []
            if self.target_encode_columns:
                extended_numeric.extend([f'{col}_target_encoded' for col in self.target_encode_columns 
                                       if f'{col}_target_encoded' in X_processed.columns])
            
            X_processed = self.feature_engineering.scale_features(X_processed, extended_numeric, fit=False)
        
        # 5. Flag clustering (using fitted transformer)
        if self.flag_transformer:
            X_processed = self.flag_transformer.transform(X_processed)
        
        logger.info("Data transformed successfully")
        return X_processed
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        
        Args:
            X (pd.DataFrame): Training data
            y (pd.Series): Target variable
            
        Returns:
            pd.DataFrame: Transformed training data
        """
        return self.fit(X, y).transform(X)
    
    def save(self, filepath):
        """
        Save the entire transformer.
        
        Args:
            filepath (str): Path to save the transformer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted transformer")
        
        transformer_data = {
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'target_encode_columns': self.target_encode_columns,
            'flag_columns': self.flag_columns,
            'missing_strategy': self.missing_strategy,
            'n_flag_clusters': self.n_flag_clusters,
            'target_col': self.target_col,
            'feature_engineering': self.feature_engineering,
            'flag_transformer': self.flag_transformer,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(transformer_data, f)
            
        logger.info(f"ComprehensiveFeatureTransformer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a fitted transformer.
        
        Args:
            filepath (str): Path to load the transformer from
            
        Returns:
            ComprehensiveFeatureTransformer: Loaded transformer
        """
        with open(filepath, 'rb') as f:
            transformer_data = pickle.load(f)
        
        transformer = cls(
            numeric_columns=transformer_data['numeric_columns'],
            categorical_columns=transformer_data['categorical_columns'],
            target_encode_columns=transformer_data['target_encode_columns'],
            flag_columns=transformer_data['flag_columns'],
            missing_strategy=transformer_data['missing_strategy'],
            n_flag_clusters=transformer_data['n_flag_clusters'],
            target_col=transformer_data['target_col']
        )
        
        transformer.feature_engineering = transformer_data['feature_engineering']
        transformer.flag_transformer = transformer_data['flag_transformer']
        transformer.is_fitted = transformer_data['is_fitted']
        
        logger.info(f"ComprehensiveFeatureTransformer loaded from {filepath}")
        return transformer
