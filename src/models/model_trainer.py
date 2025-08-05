"""
Model training and evaluation utilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training and evaluation pipeline."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y if len(y.unique()) > 1 else None
        )
    
    def train_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_name: str, hyperparams: dict = None):
        """
        Train a machine learning model.
        
        Args:
            model: Scikit-learn model
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_name (str): Name for the model
            hyperparams (dict): Hyperparameters for grid search
        """
        if hyperparams:
            # Perform hyperparameter tuning
            grid_search = GridSearchCV(
                model, hyperparams, 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        else:
            # Train with default parameters
            best_model = model
            best_model.fit(X_train, y_train)
            
        self.models[model_name] = best_model
        logger.info(f"Trained model: {model_name}")
        
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = model.score(X_test, y_test)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Add ROC AUC for binary classification
        if len(y_test.unique()) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            results['roc_auc'] = roc_auc_score(y_test, y_proba)
            
        logger.info(f"Evaluated model {model_name}: Accuracy = {accuracy:.4f}")
        
        # Track best model
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_model = model_name
            
        return results
    
    def cross_validate(self, model_name: str, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """
        Perform cross-validation.
        
        Args:
            model_name (str): Name of the model
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of folds
            
        Returns:
            dict: Cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'model_name': model_name,
            'cv_scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
        
        logger.info(f"Cross-validation for {model_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return results
    
    def save_model(self, model_name: str, file_path: str):
        """
        Save a trained model.
        
        Args:
            model_name (str): Name of the model to save
            file_path (str): Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[model_name], file_path)
        logger.info(f"Saved model {model_name} to {file_path}")
        
    def load_model(self, file_path: str, model_name: str):
        """
        Load a saved model.
        
        Args:
            file_path (str): Path to the saved model
            model_name (str): Name to assign to the loaded model
        """
        self.models[model_name] = joblib.load(file_path)
        logger.info(f"Loaded model {model_name} from {file_path}")


def get_default_models():
    """
    Get a dictionary of default models with their hyperparameters.
    
    Returns:
        dict: Dictionary of models and their hyperparameters
    """
    return {
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'hyperparams': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'hyperparams': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
    }
