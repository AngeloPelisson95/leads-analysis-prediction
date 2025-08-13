"""
Feature engineering utilities.
"""

import pandas as pd
import numpy as np
from jenkspy import JenksNaturalBreaks
import pickle
import os
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Feature engineering pipeline."""

    def __init__(self, data: pd.DataFrame, target_col: str, flag_cols: list = []):
        self.target_col = target_col
        self.flag_cols = flag_cols
        self.data = data

    def create_flag_clustering_features(self) -> pd.DataFrame:
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
            data_enhanced.groupby(feature_flag_cols)
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

    def __init__(self, n_clusters=5, target_col="leads", feature_flag_cols=None):
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

    def fit(self, X, y=None):
        """
        Fit the transformer on training data.

        Args:
            X (pd.DataFrame): Training data
            y: Not used, present for sklearn compatibility

        Returns:
            self: Returns self for method chaining
        """
        self.feature_engineering = FeatureEngineering(
            X, self.target_col, self.feature_flag_cols
        )
        logger.info("Fitting FlagClusteringTransformer...")
        logger.info(f"Using flag columns: {self.feature_flag_cols}")

        self.final_group = self.feature_engineering.create_flag_clustering_features()
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
        logger.info(
            f"Using {len(available_flag_cols)} available flag columns: {available_flag_cols}"
        )
        X_transformed["sum_flags"] = X_transformed[available_flag_cols].sum(axis=1)

        # Apply cluster mapping
        if self.cluster_mapping:
            X_transformed["flag_cluster"] = (
                X_transformed["sum_flags"].map(self.cluster_mapping).fillna(0)
            )  # Default to cluster 0 for unseen combinations
        else:
            X_transformed["flag_cluster"] = 0

        cols_to_drop = available_flag_cols + ["sum_flags"]
        logger.info(f"Dropping columns: {cols_to_drop}")
        # Drop original flag columns and sum_flags
        X_transformed = X_transformed.drop(
            columns=cols_to_drop
        )  # Remove colunas listadas em cols_to_drop
        logger.info("Dropped original flag columns and sum_flags")
        logger.info(f"Transformed data with {len(cols_to_drop)} flag columns")
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


class PreprocessingFeatures:
    """
    A comprehensive feature transformer that includes:
    - Missing value handling
    - Numerical feature scaling
    - Categorical encoding (label and target encoding)
    - Flag clustering features
    """

    def __init__(
        self,
        data: pd.DataFrame,
        location_col=None,
        fuel_type_column=None,
        cols_to_drop=None,
        outlier_columns=None,
    ):
        """
        Initialize the comprehensive transformer.

        Args:
            numeric_columns (list): columns to scale
            categorical_columns (list): columns to label encode
            target_encode_columns (list): columns to target encode
            flag_columns (list): Flag columns for clustering
            missing_strategy (str): Strategy for missing values
            n_flag_clusters (int): Number of clusters for flag clustering
            target_col (str): Target column name
        """
        self.data = data
        self.location_col = location_col
        self.fuel_type_column = fuel_type_column
        self.cols_to_drop = cols_to_drop
        self.outlier_columns = outlier_columns

    def _columns_type(self):
        self.cat_cols = []
        self.num_cols = []

        for col in self.data.columns:
            if any(
                prefix in col
                for prefix in ["cd_", "zip_", "year_", "flg_", "type", "city", "state"]
            ):
                self.cat_cols.append(col)
            else:
                self.num_cols.append(col)

        # Add priority, n_doors, and n_photos to categorical if they exist
        for col in ["priority", "n_doors", "n_photos"]:
            if col in self.data.columns and col not in self.cat_cols:
                self.cat_cols.append(col)
                if col in self.num_cols:
                    self.num_cols.remove(col)

        # Removing Target Variable from features
        self.cat_cols.remove("flg_leads") if "flg_leads" in self.cat_cols else None
        self.num_cols.remove("leads") if "leads" in self.num_cols else None
        target_cols = ["flg_leads", "leads"]

        logger.info(f"📈 Numerical features: {len(self.num_cols)}")
        logger.info(f"🏷️ Categorical features: {len(self.cat_cols)}")
        logger.info(f"🎯 Target variable: {target_cols}")

    def _location_split(self):
        """
        Create separated columns for location data state and city.

        Args:
            location_col (str): Column name containing location data.
            Ex: "STATE_CITY"

        Returns:

        """

        logger.info("Runnig location split...")

        # Creating a new column 'city' and 'state' from 'city_state'
        for _, row in self.data.iterrows():
            str_cs = row[self.location_col]
            cs_ = str_cs.split("_")
            self.data.at[_, self.location_col] = str_cs
            self.data.at[_, "city"] = cs_[1].strip()
            self.data.at[_, "state"] = cs_[0].strip()

        # Drop the original 'city_state' column
        self.data.drop(columns=["city_state"], inplace=True)

        # Function to fix encoding issues in text columns
        def fix_encoding(text):
            if isinstance(text, str):
                try:
                    return text.encode("latin1", errors="replace").decode(
                        "utf-8", errors="replace"
                    )
                except:
                    return text
            return text

        self.data["city"] = self.data["city"].apply(fix_encoding)

        return self

    def _flag_to_int(self):
        """Convert flag columns to integer type."""
        logger.info("Converting flag columns to integer type...")
        for col in self.data.columns:
            if col.startswith("flg_"):
                self.data[col] = self.data[col].fillna(0)
                self.data[col] = (
                    self.data[col]
                    .replace({"S": "Y"})
                    .replace({"Y": 1, "N": 0})
                    .astype(int)
                )

        logger.info("Flag columns converted to integer type successfully")
        return self

    def _fuel_type_to_flag(self):
        """Convert fuel type column to flag columns.
        Args:
            fuel_type_col (str): Column name containing fuel type data.
        Returns:
            pd.DataFrame: DataFrame with new flag columns for each fuel type.
        """
        logger.info("Converting fuel type to flag columns...")

        fuel_types = []
        for item in self.data[self.fuel_type_column].unique():
            if isinstance(item, str):  # check if item is a string
                # Split the string by spaces and commas, and strip any whitespace
                str_lst = item.split(" ")
                for i in range(len(str_lst)):
                    str_lst[i] = str_lst[i].strip().replace(",", "")  # remove commas
                    if str_lst[i] != "e":  # check if the item is not 'e'
                        if (
                            str_lst[i] not in fuel_types
                        ):  # check if the item is not already in the list
                            fuel_types.append(str_lst[i])

        logger.info(f"Found {len(fuel_types)} unique fuel types: {fuel_types}")
        if "natural" in fuel_types:
            fuel_types.remove("natural")
        if "gas" in fuel_types:
            fuel_types.remove("gas")
            fuel_types.append("gas natural")

        self.data["fuel_type"] = self.data["fuel_type"].fillna(
            ""
        )  # Fill NaN values with empty string to avoid errors

        # Third, we need to create a new column for each fuel type
        for fuel in fuel_types:
            self.data[f"flg_{fuel.replace(" ", "_")}"] = self.data["fuel_type"].apply(
                lambda x: 1 if fuel in x else 0 if x == "" else 0
            )

        # Now we can drop the original fuel_type column
        self.data.drop(columns=[self.fuel_type_column], inplace=True)

        return self

    def _remove_duplicated(self):
        """Remove duplicate columns from the DataFrame."""
        logger.info("Removing duplicate columns...")
        # Identify duplicate columns
        duplicate_columns = self.data.columns[self.data.columns.duplicated()].tolist()
        if duplicate_columns:
            logger.warning(f"Found duplicate columns: {duplicate_columns}")
            # Remove duplicate columns, keeping the first occurrence
            self.data = self.data.loc[:, ~self.data.columns.duplicated()]
            logger.info("Duplicate columns removed successfully")
        else:
            logger.info("No duplicate columns found")

        self.data = self.data.drop_duplicates().reset_index(drop=True)

        return self

    def _clean_spurious_values(self):
        """Clean spurious values in the DataFrame."""
        logger.info("Cleaning spurious values...")
        rows_with_minus_one = (self.data == -1).any(axis=1).sum()
        if rows_with_minus_one > 0:
            logger.warning(
                f"Found {rows_with_minus_one} rows with -1 values, removing them"
            )
        else:
            logger.info("No spurious -1 values found")
        self.data = self.data[~(self.data == -1).any(axis=1)].reset_index(drop=True)

        return self

    def _remove_outliers_iqr(self, lower_percentile=0.01, upper_percentile=0.99, k=3):
        """Remove outliers using percentile-based approach"""

        columns = self.outlier_columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.k = k

        for col in columns:
            q1 = self.data[col].quantile(self.lower_percentile)
            q3 = self.data[col].quantile(self.upper_percentile)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            outliers_low = (self.data[col] < lower).sum()
            outliers_high = (self.data[col] > upper).sum()
            total_outliers = outliers_low + outliers_high

            logger.info(f"{col}:")
            logger.info(
                f"  - Lower bound ({self.lower_percentile:.1%}): {lower:.2f} ({outliers_low} outliers)"
            )
            logger.info(
                f"  - Upper bound ({self.upper_percentile:.1%}): {upper:.2f} ({outliers_high} outliers)"
            )
            logger.info(
                f"  - Total outliers: {total_outliers} ({total_outliers/len(self.data)*100:.2f}%)"
            )

            self.data = self.data[(self.data[col] >= lower) & (self.data[col] <= upper)]
            logger.info(f"  - Rows removed: {len(self.data) - len(self.data)}n")

        return self

    def _feature_setup(self):
        """Setup features for the transformer."""
        logger.info("Setting up features...")

        # Handle missing values
        self.processed_data = self.data.copy()

        # Only drop columns that are present in processed_data
        cols_to_drop_present = [
            col for col in self.cols_to_drop if col in self.processed_data.columns
        ]
        self.input_data = self.processed_data.drop(columns=cols_to_drop_present)
        self.input_columns = self.input_data.columns.tolist()
        self.data = self.input_data.copy()
        logger.info("Feature setup completed successfully")
        return self


class PreprocessingFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, location_col, fuel_type_column, cols_to_drop, outlier_columns):
        self.location_col = location_col
        self.fuel_type_column = fuel_type_column
        self.cols_to_drop = cols_to_drop
        self.outlier_columns = outlier_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        self.pre = PreprocessingFeatures(
            data=data,
            location_col=self.location_col,
            fuel_type_column=self.fuel_type_column,
            cols_to_drop=self.cols_to_drop,
            outlier_columns=self.outlier_columns,
        )

        self.pre._columns_type()
        self.pre._location_split()
        self.pre._flag_to_int()
        self.pre._fuel_type_to_flag()
        self.pre._remove_duplicated()
        self.pre._clean_spurious_values()
        self.pre._remove_outliers_iqr()
        self.pre._feature_setup()

        return self.pre.data

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
