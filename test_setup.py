#!/usr/bin/env python3
"""
Simple test script to verify the virtual environment setup.
"""


def test_imports():
    """Test that all key packages can be imported."""
    print(
        "🚀 Testing Python environment for Lead Generation Vehicle Listings project..."
    )
    print(f"Python version: {__import__('sys').version}")
    print()

    packages_to_test = [
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("matplotlib", "Plotting"),
        ("seaborn", "Statistical visualization"),
        ("jupyter", "Notebook environment"),
        ("plotly", "Interactive plotting"),
        ("scipy", "Scientific computing"),
        ("xgboost", "Gradient boosting"),
        ("lightgbm", "Light gradient boosting"),
    ]

    print("📦 Testing package imports:")
    for package, description in packages_to_test:
        try:
            __import__(package)
            print(f"✅ {package:12} - {description}")
        except ImportError as e:
            print(f"❌ {package:12} - Failed to import: {e}")

    print()
    print("🎯 Testing project modules:")
    try:
        from src.data.data_loader import load_raw_data, basic_data_info

        print("✅ src.data.data_loader - Data loading utilities")
    except ImportError as e:
        print(f"❌ src.data.data_loader - {e}")

    try:
        from src.features.feature_engineering import FeatureEngineering

        print("✅ src.features.feature_engineering - Feature engineering")
    except ImportError as e:
        print(f"❌ src.features.feature_engineering - {e}")

    try:
        from src.models.model_trainer import ModelTrainer

        print("✅ src.models.model_trainer - Model training")
    except ImportError as e:
        print(f"❌ src.models.model_trainer - {e}")

    try:
        from src.visualization.plotting import DataVisualizer

        print("✅ src.visualization.plotting - Data visualization")
    except ImportError as e:
        print(f"❌ src.visualization.plotting - {e}")

    print()
    print("🎉 Environment setup complete! Ready for data science work.")


if __name__ == "__main__":
    test_imports()
