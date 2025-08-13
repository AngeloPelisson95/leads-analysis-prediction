"""
Model training and evaluation utilities for lead prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve,
    KFold,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
import optuna.integration.lightgbm as lgb
import optuna
from sklearn.base import clone
import joblib
import pickle
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class LeadsPredictionTrainer:
    """Lead prediction model trainer with LightGBM and Optuna optimization."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline = None
        self.best_params = None
        self.study = None
        self.trained_model = None

        # Features to drop (based on analysis)
        self.feat_to_drop = [
            "cd_type_individual",
            "cd_advertise",
            "cd_client",
            "flg_rain_sensor",
            "flg_diesel",
            "flg_eletrico",
            "flg_benzina",
            "flg_pcd",
            "flg_trade_in",
            "flg_armored",
            "flg_factory_warranty",
            "flg_all_dealership_schedule_vehicle",
            "flg_all_dealership_services",
            "flg_single_owner",
            "priority",
            "cd_model_vehicle",
            "cd_version_vehicle",
            "flg_lincese",
            "flg_tax_paid",
            "n_doors",
            "flg_alloy_wheels",
            "flg_gas_natural",
        ]

    def create_pipeline(self):
        """Create the complete ML pipeline."""
        self.pipeline = Pipeline(
            steps=[
                (
                    "encoding",
                    ColumnTransformer(
                        transformers=[
                            (
                                "city",
                                TargetEncoder(
                                    cols=["city"], smoothing=0.5, min_samples_leaf=1000
                                ),
                                ["city"],
                            ),
                            (
                                "state",
                                TargetEncoder(
                                    cols=["state"], smoothing=5, min_samples_leaf=500
                                ),
                                ["state"],
                            ),
                            (
                                "scaler",
                                StandardScaler(),
                                [
                                    "views",
                                    "phone_clicks",
                                    "vl_advertise",
                                    "km_vehicle",
                                    "vl_market",
                                ],
                            ),
                            (
                                "transmission",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value", unknown_value=-1
                                ),
                                ["transmission_type"],
                            ),
                        ],
                        remainder="passthrough",
                    ),
                ),
                ("model", LGBMRegressor(random_state=self.random_state, verbosity=-1)),
            ]
        )
        return self.pipeline

    def prepare_data(self, df_processed):
        """Prepare features and target from processed dataframe."""
        X = df_processed.drop("leads", axis=1)
        y = df_processed["leads"]
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Split data into training and testing sets."""
        return train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

    def optimize_hyperparameters(self, X, y, X_train, y_train):
        """Optimize hyperparameters using Optuna."""
        if self.pipeline is None:
            self.create_pipeline()

        # Prepare encoding step
        prep = clone(self.pipeline.named_steps["encoding"])

        # Fit the preprocessing pipeline on training data and transform both sets
        prep.fit(X, y)  # Learn encoding parameters from training data only
        X_tr = prep.transform(X_train)  # Transform training features
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        dtrain = lgb.Dataset(X_tr, label=y_train)

        params = {
            "objective": "regression",
            "metric": "mean_squared_error",
            "verbosity": -1,
            "boosting_type": "gbdt",
        }

        self.tuner = lgb.LightGBMTunerCV(params, dtrain, folds=kf)

        self.tuner.run()
        self.best_params = self.tuner.best_params
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best MSE: {self.study.best_value:.4f}")

        return self.best_params

    def train_final_model(self, X_train, y_train):
        """Train final model with best parameters."""
        if self.best_params is None:
            raise ValueError("Must run hyperparameter optimization first")

        if self.pipeline is None:
            self.create_pipeline()

        # Set best parameters
        self.pipeline.named_steps["model"].set_params(**self.best_params)

        # Train
        self.pipeline.fit(X_train, y_train)
        self.trained_model = self.pipeline

        logger.info("Final model trained successfully")
        return self.trained_model

    def evaluate_model(self, X_test, y_test):
        """Evaluate trained model."""
        if self.trained_model is None:
            raise ValueError("Must train model first")

        y_pred = self.trained_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results = {"mse": mse, "rmse": rmse, "r2": r2, "predictions": y_pred}

        logger.info(f"Model evaluation - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return results

    def calculate_baseline(self, y_train):
        """Calculate baseline performance."""
        baseline_pred = np.full_like(y_train, y_train.mean(), dtype=float)
        baseline_mse = np.mean((y_train - baseline_pred) ** 2)
        logger.info(f"Baseline MSE: {baseline_mse:.4f}")
        return baseline_mse

    def plot_learning_curve(self, X_train, y_train, cv=5):
        """Plot learning curve to assess overfitting."""
        if self.trained_model is None:
            raise ValueError("Must train model first")

        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.trained_model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=-1,
            shuffle=True,
            random_state=self.random_state,
            return_times=False,
        )

        train_rmse = -train_scores
        test_rmse = -test_scores

        train_mean = train_rmse.mean(axis=1)
        train_std = train_rmse.std(axis=1)
        test_mean = test_rmse.mean(axis=1)
        test_std = test_rmse.std(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, "o-", label="Train RMSE")
        plt.plot(train_sizes, test_mean, "o-", label="Validation RMSE")
        plt.fill_between(
            train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15
        )
        plt.fill_between(
            train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15
        )
        plt.xlabel("Training Set Size")
        plt.ylabel("RMSE")
        plt.title("Learning Curve (RMSE)")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Check for overfitting
        gap = test_mean - train_mean
        gap_ratio = gap / np.maximum(train_mean, 1e-9)
        last_ratio = gap_ratio[-1]

        if last_ratio > 0.15:
            print(
                f"⚠️ Possible overfitting detected: gap={gap[-1]:.3f}, gap/train={last_ratio:.1%} (> 15%)"
            )
        else:
            print(
                f"✅ No overfitting signs: gap={gap[-1]:.3f}, gap/train={last_ratio:.1%} (≤ 15%)"
            )

        return {
            "train_sizes": train_sizes,
            "train_scores": train_rmse,
            "test_scores": test_rmse,
            "overfitting_detected": last_ratio > 0.15,
        }

    def plot_predictions(self, y_test, y_pred):
        """Plot real vs predicted values."""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, s=20, edgecolor=None)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        plt.plot(lims, lims, "r--", label="Ideal")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_residuals(self, y_test, y_pred):
        """Plot residuals distribution."""
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 4))
        sns.histplot(residuals, bins=40, kde=True)
        plt.title("Residuals Distribution (y_true - y_pred)")
        plt.xlabel("Residual")
        plt.grid(True)
        plt.show()

        return residuals

    def predict(self, X):
        """Make predictions on new data."""
        if self.trained_model is None:
            raise ValueError("Must train model first")
        return self.trained_model.predict(X)

    def predict_with_categories(self, X):
        """Make predictions and categorize into business ranges."""
        predictions = self.predict(X)

        categories = []
        for pred in predictions:
            if pred <= 5:
                categories.append("Low Performance")
            elif pred <= 15:
                categories.append("Moderate Performance")
            elif pred <= 30:
                categories.append("High Performance")
            else:
                categories.append("Exceptional Performance")

        return predictions, categories

    def save_model(self, file_path: str):
        """Save trained model."""
        if self.trained_model is None:
            raise ValueError("Must train model first")

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_model, file_path)
        logger.info(f"Model saved to {file_path}")

    def save_study(self, file_path: str):
        """Save Optuna study."""
        if self.tuner is None:
            raise ValueError("Must run optimization first")

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.tuner, f)
        logger.info(f"Study saved to {file_path}")

    def load_model(self, file_path: str):
        """Load saved model."""
        self.trained_model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")

    def load_study(self, file_path: str):
        """Load saved study."""
        with open(file_path, "rb") as f:
            self.study = pickle.load(f)
        if self.study is None:
            raise ValueError(
                "Loaded study is None. Please check the file or run optimization first."
            )
        self.best_params = self.study.best_params.copy()
        self.best_params["verbosity"] = -1
        self.best_params.pop("early_stopping_rounds", None)
        logger.info(f"Study loaded from {file_path}")


def get_required_columns():
    """
    Get list of required columns for model input.

    Returns:
        list: Required column names for raw data
    """
    return [
        # Target
        "leads",
        # Features that will be kept after preprocessing
        "views",
        "phone_clicks",
        "cd_vehicle_brand",
        "year_model",
        "zip_2dig",
        "vl_advertise",
        "n_photos",
        "km_vehicle",
        "vl_market",
        "transmission_type",
        "flg_leather_seats",
        "flg_parking_sensor",
        "city_state",
        # Flag columns for clustering
        "flg_electric_locks",
        "flg_air_conditioning",
        "flg_electric_windows",
        "flg_rear_defogger",
        "flg_heater",
        "flg_alarm",
        "flg_airbag",
        "flg_abs",
        # Fuel type for preprocessing
        "fuel_type",
    ]


def create_sample_data():
    """
    Create sample data structure for testing.

    Returns:
        pd.DataFrame: Sample dataframe with correct structure
    """
    sample_data = {
        "leads": [10, 25, 5, 15, 30],
        "views": [150, 300, 80, 200, 400],
        "phone_clicks": [12, 25, 6, 18, 35],
        "cd_vehicle_brand": [1, 2, 3, 1, 2],
        "year_model": [2020, 2019, 2021, 2018, 2022],
        "zip_2dig": [1, 2, 3, 1, 2],
        "vl_advertise": [45000, 60000, 35000, 50000, 70000],
        "n_photos": [8, 6, 10, 7, 9],
        "km_vehicle": [30000, 50000, 15000, 40000, 20000],
        "vl_market": [48000, 58000, 37000, 52000, 68000],
        "transmission_type": [
            "automatico",
            "semi automatico",
            "manual",
            "automatico",
            "manual",
        ],
        "flg_leather_seats": [1, 1, 0, 1, 1],
        "flg_parking_sensor": [1, 1, 0, 0, 1],
        "city_state": [
            "SP_São Paulo",
            "RJ_Rio de Janeiro",
            "MG_Belo Horizonte",
            "SP_São Paulo",
            "PR_Curitiba",
        ],
        "flg_gasolina": [1, 0, 1, 1, 0],
        "flg_electric_locks": [1, 1, 0, 1, 1],
        "flg_air_conditioning": [1, 1, 1, 1, 1],
        "flg_electric_windows": [1, 1, 0, 1, 1],
        "flg_rear_defogger": [1, 1, 1, 1, 1],
        "flg_heater": [1, 1, 1, 1, 1],
        "flg_alarm": [1, 1, 0, 1, 1],
        "flg_airbag": [1, 1, 1, 1, 1],
        "flg_abs": [1, 1, 1, 1, 1],
        "flg_alcool": [0, 1, 0, 0, 1],
        "fuel_type": ["gasolina", "gasolina e alcool", "gasolina", "alcool", "flex"],
    }

    return pd.DataFrame(sample_data)
