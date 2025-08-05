"""
Feature engineering utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Feature engineering pipeline."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            strategy (str): Strategy for handling missing values
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        data_clean = data.copy()
        
        for column in data_clean.columns:
            if data_clean[column].isnull().sum() > 0:
                if data_clean[column].dtype in ['float64', 'int64']:
                    if strategy == 'mean':
                        data_clean[column].fillna(data_clean[column].mean(), inplace=True)
                    elif strategy == 'median':
                        data_clean[column].fillna(data_clean[column].median(), inplace=True)
                    elif strategy == 'mode':
                        data_clean[column].fillna(data_clean[column].mode()[0], inplace=True)
                else:
                    data_clean[column].fillna(data_clean[column].mode()[0], inplace=True)
                    
        logger.info(f"Handled missing values using {strategy} strategy")
        return data_clean
    
    def encode_categorical_variables(self, data: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): List of columns to encode
            
        Returns:
            pd.DataFrame: Data with encoded categorical variables
        """
        data_encoded = data.copy()
        
        if columns is None:
            columns = data_encoded.select_dtypes(include=['object']).columns
            
        for column in columns:
            if column in data_encoded.columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    data_encoded[column] = self.label_encoders[column].fit_transform(data_encoded[column])
                else:
                    data_encoded[column] = self.label_encoders[column].transform(data_encoded[column])
                    
        logger.info(f"Encoded categorical variables: {columns}")
        return data_encoded
    
    def scale_features(self, data: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list): List of columns to scale
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        data_scaled = data.copy()
        
        if columns is None:
            columns = data_scaled.select_dtypes(include=['float64', 'int64']).columns
            
        if len(columns) > 0:
            data_scaled[columns] = self.scaler.fit_transform(data_scaled[columns])
            logger.info(f"Scaled features: {list(columns)}")
            
        return data_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """
        Select top k features based on statistical tests.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Selected features
        """
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        selected_features = X.columns[self.feature_selector.get_support()]
        logger.info(f"Selected {len(selected_features)} features: {list(selected_features)}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
