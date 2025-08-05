# Lead Generation Analysis and Optimization in Vehicle Listings
## Overview
This project aims to analyze vehicle listing data and understand which factors influence lead generation — that is, expressions of interest from potential buyers. Based on this analysis, the project develops data-driven strategies and predictive models to improve ad performance and maximize engagement.

## Objectives
Perform exploratory data analysis (EDA) to identify patterns related to the number of leads.

Evaluate the impact of features such as brand, model, city, price, and vehicle year.

Build a predictive model to estimate the expected number of leads.

Propose practical alternatives to enhance lead generation for future listings.



## Project Structure

```
├── README.md              # Project overview and instructions
├── setup.py               # Package setup and dependencies
├── requirements.txt       # Python package requirements
├── .gitignore            # Git ignore rules
├── .env.example          # Environment variables template
│
├── config/               # Configuration files
│   └── config.yaml       # Project configuration
│
├── data/                 # Data directories
│   ├── raw/              # Original, immutable data
│   ├── interim/          # Intermediate data transformations
│   ├── processed/        # Final data for modeling
│   └── external/         # External data sources
│
├── docs/                 # Project documentation
│   └── Case Cientista de dados_1_semana.pdf
│
├── notebooks/            # Jupyter notebooks
│   ├── exploratory/      # Data exploration notebooks
│   └── modeling/         # Model development notebooks
│
├── src/                  # Source code
│   ├── __init__.py
│   ├── data/             # Data loading and processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   └── visualization/    # Plotting and visualization
│
├── reports/              # Generated analysis reports
│   └── figures/          # Generated graphics and figures
│
└── tests/                # Unit tests
    └── test_*.py
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lead-generation-vehicle-listings
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` file with your specific configurations.

## Usage

### Data Processing

```python
from src.data.data_loader import load_raw_data, basic_data_info

# Load data
data = load_raw_data('data/raw/Case 1 - dados.csv')

# Get basic information
info = basic_data_info(data)
print(info)
```

### Feature Engineering

```python
from src.features.feature_engineering import FeatureEngineering

fe = FeatureEngineering()

# Handle missing values
data_clean = fe.handle_missing_values(data)

# Encode categorical variables
data_encoded = fe.encode_categorical_variables(data_clean)

# Scale features
data_scaled = fe.scale_features(data_encoded)
```

### Model Training

```python
from src.models.model_trainer import ModelTrainer, get_default_models

trainer = ModelTrainer()

# Split data
X_train, X_test, y_train, y_test = trainer.split_data(X, y)

# Get default models
models = get_default_models()

# Train models
for name, config in models.items():
    trainer.train_model(
        config['model'], 
        X_train, y_train, 
        name, 
        config['hyperparams']
    )
    
    # Evaluate model
    results = trainer.evaluate_model(name, X_test, y_test)
    print(f"{name}: {results['accuracy']:.4f}")
```

### Visualization

```python
from src.visualization.plotting import DataVisualizer

viz = DataVisualizer()

# Plot distributions
viz.plot_distribution(data, 'column_name', 'distribution_plot')

# Plot correlation matrix
viz.plot_correlation_matrix(data, 'correlation_matrix')
```

## Notebooks

The project includes several Jupyter notebooks:

- **Exploratory Analysis**: Located in `notebooks/exploratory/`
  - Data exploration and initial analysis
  - Statistical summaries and visualizations
  
- **Model Development**: Located in `notebooks/modeling/`
  - Model training and evaluation
  - Hyperparameter tuning
  - Performance comparison

## Testing

Run tests using pytest:

```bash
pytest tests/
```

For coverage report:

```bash
pytest --cov=src tests/
```

## Project Guidelines

### Data Management
- **Raw data**: Never modify files in `data/raw/`
- **Processed data**: Store cleaned datasets in `data/processed/`
- **Interim data**: Use `data/interim/` for intermediate transformations

### Code Organization
- Keep functions small and focused
- Use type hints where appropriate
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes

### Version Control
- Commit frequently with meaningful messages
- Clear notebook outputs before committing
- Use `.gitignore` to exclude unnecessary files

### Documentation
- Update README when adding new features
- Document any data assumptions or limitations
- Include examples in docstrings

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact Angelo Pelisson.
