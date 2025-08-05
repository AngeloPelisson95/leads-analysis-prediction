"""
Data visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DataVisualizer:
    """Data visualization utilities."""
    
    def __init__(self, figsize: tuple = (12, 8), save_path: str = "reports/figures"):
        self.figsize = figsize
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def plot_distribution(self, data: pd.DataFrame, column: str, save_name: str = None):
        """
        Plot distribution of a variable.
        
        Args:
            data (pd.DataFrame): Input data
            column (str): Column to plot
            save_name (str): Name to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(data[column], bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title(f'Distribution of {column}')
        ax1.set_xlabel(column)
        ax1.set_ylabel('Frequency')
        
        # Box plot
        ax2.boxplot(data[column])
        ax2.set_title(f'Box Plot of {column}')
        ax2.set_ylabel(column)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {save_name}.png")
        
        plt.show()
        
    def plot_correlation_matrix(self, data: pd.DataFrame, save_name: str = None):
        """
        Plot correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Input data
            save_name (str): Name to save the plot
        """
        # Calculate correlation matrix
        corr = data.select_dtypes(include=[np.number]).corr()
        
        # Create heatmap
        plt.figure(figsize=self.figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation matrix to {save_name}.png")
        
        plt.show()
        
    def plot_target_distribution(self, data: pd.DataFrame, target_column: str, save_name: str = None):
        """
        Plot target variable distribution.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            save_name (str): Name to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Count plot
        if data[target_column].dtype == 'object' or data[target_column].nunique() < 10:
            sns.countplot(data=data, x=target_column)
            plt.title(f'Distribution of {target_column}')
            plt.xticks(rotation=45)
        else:
            plt.hist(data[target_column], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {target_column}')
            plt.xlabel(target_column)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved target distribution to {save_name}.png")
        
        plt.show()
        
    def plot_feature_importance(self, importance_scores: dict, save_name: str = None):
        """
        Plot feature importance.
        
        Args:
            importance_scores (dict): Dictionary of feature names and importance scores
            save_name (str): Name to save the plot
        """
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        plt.figure(figsize=self.figsize)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_name}.png")
        
        plt.show()
        
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list, save_name: str = None):
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            class_names (list): List of class names
            save_name (str): Name to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_name}.png")
        
        plt.show()
        
    def plot_model_comparison(self, results: list, save_name: str = None):
        """
        Plot model comparison chart.
        
        Args:
            results (list): List of model evaluation results
            save_name (str): Name to save the plot
        """
        model_names = [result['model_name'] for result in results]
        accuracies = [result['accuracy'] for result in results]
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(model_names, accuracies, alpha=0.7)
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_path / f"{save_name}.png", dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison to {save_name}.png")
        
        plt.show()
