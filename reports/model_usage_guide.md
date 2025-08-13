# 📖 Lead Prediction Model Usage Guide

## 🎯 Overview

This guide explains how to use the LightGBM-trained lead prediction model to predict the number of leads a vehicle advertisement will generate.

## 📋 Required Data Structure

### Required Columns (24 columns):

```python
required_columns = [
    # Engagement metrics
    'views',                    # Number of views
    'phone_clicks',             # Number of phone clicks
    
    # Vehicle information
    'cd_vehicle_brand',         # Brand code (number)
    'year_model',               # Model year
    'vl_advertise',             # Advertised value
    'km_vehicle',               # Mileage
    'vl_market',                # Market value
    'transmission_type',        # Transmission type ('Manual'/'Automatic')
    'fuel_type',                # Fuel type ('Gasoline'/'Flex'/etc)
    
    # Advertisement information
    'n_photos',                 # Number of photos
    'zip_2dig',                 # First 2 digits of ZIP code
    'city_state',               # City/State format 'City/State'
    
    # Vehicle characteristics (flags 0/1)
    'flg_leather_seats',        # Leather seats
    'flg_parking_sensor',       # Parking sensor
    'flg_gasolina',             # Gasoline fuel
    'flg_electric_locks',       # Electric locks
    'flg_air_conditioning',     # Air conditioning
    'flg_electric_windows',     # Electric windows
    'flg_rear_defogger',        # Rear defogger
    'flg_heater',               # Heater
    'flg_alarm',                # Alarm
    'flg_airbag',               # Airbag
    'flg_abs',                  # ABS brakes
    'flg_alcool',               # Alcohol fuel
]
```

## 🚀 How to Use

### 1. Loading the Model

```python
import joblib
import pandas as pd
import numpy as np

# Load saved pipelines
ml_pipeline = joblib.load('models/complete_ml_pipeline.joblib')
prep_pipeline = joblib.load('models/preprocessing_pipeline.joblib')
```

### 2. Prepare Input Data

```python
# Example input data
new_data = pd.DataFrame({
    'views': [150, 300, 80],
    'phone_clicks': [12, 25, 6],
    'cd_vehicle_brand': [1, 2, 1],
    'year_model': [2020, 2019, 2021],
    'zip_2dig': [1, 2, 1],
    'vl_advertise': [45000, 60000, 35000],
    'n_photos': [8, 6, 10],
    'km_vehicle': [30000, 50000, 15000],
    'vl_market': [48000, 58000, 37000],
    'transmission_type': ['Manual', 'Automatic', 'Manual'],
    'fuel_type': ['Gasoline', 'Flex', 'Gasoline'],
    'flg_leather_seats': [1, 1, 0],
    'flg_parking_sensor': [1, 1, 0],
    'city_state': ['São Paulo/SP', 'Rio de Janeiro/RJ', 'Belo Horizonte/MG'],
    'flg_gasolina': [1, 0, 1],
    'flg_electric_locks': [1, 1, 0],
    'flg_air_conditioning': [1, 1, 1],
    'flg_electric_windows': [1, 1, 0],
    'flg_rear_defogger': [1, 1, 1],
    'flg_heater': [1, 1, 1],
    'flg_alarm': [1, 1, 0],
    'flg_airbag': [1, 1, 1],
    'flg_abs': [1, 1, 1],
    'flg_alcool': [0, 1, 0]
})
```

### 3. Make Predictions

```python
def predict_leads(new_data):
    # Apply preprocessing
    processed_data = prep_pipeline.transform(new_data)
    
    # Remove target column if it exists
    if 'leads' in processed_data.columns:
        X = processed_data.drop('leads', axis=1)
    else:
        X = processed_data
    
    # Make predictions
    predictions = ml_pipeline.predict(X)
    
    # Categorize results
    categories = []
    for pred in predictions:
        if pred <= 5:
            categories.append("🔴 Low Performance")
        elif pred <= 15:
            categories.append("🟡 Moderate Performance")
        elif pred <= 30:
            categories.append("🟢 High Performance")
        else:
            categories.append("🌟 Exceptional Performance")
    
    return predictions, categories

# Use the function
predictions, categories = predict_leads(new_data)

# Show results
results = pd.DataFrame({
    'Predicted_Leads': predictions,
    'Performance_Category': categories
})
print(results)
```

## 📊 Interpreting Results

### Performance Categories:

| Lead Range | Category | Recommended Action |
|------------|----------|-------------------|
| 0-5 leads | 🔴 **Low Performance** | Review ad quality, adjust price |
| 6-15 leads | 🟡 **Moderate Performance** | Standard monitoring, minor adjustments |
| 16-30 leads | 🟢 **High Performance** | Replicate success factors |
| 31+ leads | 🌟 **Exceptional Performance** | Maximum investment, case study |

## 🔧 Complete Usage Example

```python
from src.models.model_trainer import LeadsPredictionTrainer, create_sample_data
import pandas as pd

# Create sample data
sample_data = create_sample_data()

# Or load real data
# raw_data = pd.read_csv('your_data.csv')

# Use trained model (assuming it's already loaded)
def business_prediction_workflow(data):
    """Complete workflow for business prediction"""
    
    # 1. Make predictions
    predictions, categories = predict_leads(data)
    
    # 2. Create report
    report = pd.DataFrame({
        'Ad_ID': range(1, len(predictions) + 1),
        'Predicted_Leads': predictions.round(1),
        'Performance_Category': categories,
        'Views': data['views'],
        'Phone_Clicks': data['phone_clicks'],
        'Vehicle_Value': data['vl_advertise'],
        'Photos': data['n_photos']
    })
    
    # 3. Summary statistics
    print("📊 PREDICTION SUMMARY:")
    print(f"Total ads analyzed: {len(predictions)}")
    print(f"Average predicted leads: {predictions.mean():.1f}")
    print(f"Total estimated leads: {predictions.sum():.0f}")
    
    print("\n📈 Distribution by category:")
    category_counts = report['Performance_Category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(report)) * 100
        print(f"  {category}: {count} ads ({percentage:.1f}%)")
    
    return report

# Run analysis
final_report = business_prediction_workflow(new_data)
print("\n📋 FINAL REPORT:")
print(final_report)
```

## ⚠️ Important Validations

### Check Data Quality:

```python
def validate_input_data(data):
    """Validate input data"""
    errors = []
    
    # Check required columns
    required_cols = get_required_columns()  # From model_trainer.py
    missing_cols = set(required_cols) - set(data.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check negative values where they don't make sense
    numeric_positive = ['views', 'phone_clicks', 'vl_advertise', 'km_vehicle', 'n_photos']
    for col in numeric_positive:
        if col in data.columns and (data[col] < 0).any():
            errors.append(f"Negative values in {col}")
    
    # Check flags (should be 0 or 1)
    flag_cols = [col for col in data.columns if col.startswith('flg_')]
    for col in flag_cols:
        if not data[col].isin([0, 1]).all():
            errors.append(f"Flag {col} must be 0 or 1")
    
    # Check city_state format
    if 'city_state' in data.columns:
        invalid_format = ~data['city_state'].str.contains('/', na=False)
        if invalid_format.any():
            errors.append("city_state must have format 'City/State'")
    
    return errors

# Use validation
errors = validate_input_data(new_data)
if errors:
    print("❌ Errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✅ Data valid for prediction")
```

## 🎯 Model Metrics

- **RMSE**: ~6.92 leads
- **R²**: ~70.3% (70% of variance explained)
- **Improvement over baseline**: ~65%
- **Prediction time**: <1ms per sample

## 📞 Support

For questions or issues:
1. Check if all required columns are present
2. Confirm data format (flags 0/1, city_state with '/')
3. Validate data types (numeric where expected)
4. Check error logs for debugging

---

**Model Version**: 1.0  
**Last Update**: August 12, 2025  
**Algorithm**: LightGBM with Optuna optimization
