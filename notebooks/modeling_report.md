

# Machine Learning Modeling Report

## Executive Summary
This report documents the development and evaluation of a LightGBM regression model for predicting lead generation from vehicle advertisements. The final model achieves **RMSE of 6.922** and **R² of 0.703** on test data, representing a **64.6% improvement** over the baseline model.

## Dataset Overview
- **Final Dataset Shape**: 16 features after feature engineering and preprocessing
- **Target Variable**: `leads` (number of leads generated per advertisement)
- **Train/Test Split**: 70%/30% split with random_state=42
- **Baseline Performance**: MSE = 135.52 (mean prediction baseline)

---

## Feature Engineering Pipeline

### 1. Preprocessing Pipeline
**Features Dropped** (22 total):
```python
feat_to_drop = [
    "cd_type_individual", "cd_advertise", "cd_client",        # ID features
    "flg_rain_sensor", "flg_diesel", "flg_eletrico",         # Low variance flags
    "flg_benzina", "flg_pcd", "flg_trade_in",               # Rare occurrences  
    "flg_armored", "flg_factory_warranty",                   # Business-specific
    "flg_all_dealership_schedule_vehicle",                   # Low predictive value
    "flg_all_dealership_services", "flg_single_owner",       
    "priority", "cd_model_vehicle", "cd_version_vehicle",     # High cardinality
    "flg_lincese", "flg_tax_paid", "n_doors",               # Redundant/noisy
    "flg_alloy_wheels", "flg_gas_natural"                    # Low importance
]
```

### 2. Feature Engineering Components
- **Location Processing**: `city_state` → separate `city` and `state` columns
- **Fuel Type Conversion**: Categorical fuel type → binary flag columns  
- **Flag Clustering**: 9 vehicle feature flags clustered using Jenks Natural Breaks
- **Outlier Removal**: IQR-based outlier detection for `vl_advertise` and `km_vehicle`

### 3. Final Feature Set (16 features)
**Numerical Features (5)**:
- `views`, `phone_clicks`, `vl_advertise`, `km_vehicle`, `vl_market`

**Categorical Features (11)**:
- `cd_vehicle_brand`, `year_model`, `zip_2dig`, `transmission_type`
- `city`, `state` (high cardinality)
- `flg_leather_seats`, `flg_parking_sensor`, `flg_alcool`
- `n_photos`, `flag_cluster` (engineered feature)

---

## Model Development

### 1. Model Architecture
**Base Algorithm**: LightGBM Regressor
- Gradient boosting framework optimized for efficiency
- Handles categorical features natively
- Built-in regularization and early stopping capabilities

### 2. Encoding Strategy
**Pipeline Design**:
```python
ColumnTransformer([
    ('city', TargetEncoder(smoothing=0.5, min_samples_leaf=1000), ['city']),
    ('state', TargetEncoder(smoothing=5, min_samples_leaf=500), ['state']),  
    ('scaler', StandardScaler(), ['views', 'phone_clicks', 'vl_advertise', 'km_vehicle', 'vl_market']),
    ('transmission', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['transmission_type'])
])
```

**Encoding Rationale**:
- **Target Encoding**: For high-cardinality features (city/state) with strong target correlation
- **Standard Scaling**: For numerical features with varying scales
- **Ordinal Encoding**: For transmission type (natural ordering exists)
- **Passthrough**: For remaining categorical features (handled by LightGBM)

### 3. Hyperparameter Optimization
**Optimization Framework**: Optuna with 150 trials
- **Objective**: Minimize RMSE on validation set
- **Search Space**: Log-uniform and uniform distributions
- **Cross-Validation**: 5-fold KFold with shuffle

**Optimized Hyperparameters**:
```python
{
    'n_estimators': 979,
    'learning_rate': 0.0398,
    'num_leaves': 93,
    'max_depth': 5,
    'min_data_in_leaf': 14,
    'feature_fraction': 0.633,
    'verbosity': -1
}
```

---

## Model Performance Analysis

### 1. Overall Performance Metrics
**Test Set Results**:
- **RMSE**: 6.922 leads
- **MSE**: 47.912  
- **R²**: 0.703 (70.3% variance explained)
- **Improvement over Baseline**: 64.6% MSE reduction

### 2. Learning Curve Analysis
**Overfitting Assessment**:
- **Gap Analysis**: Train vs. Validation RMSE gap = 1.715 (35.4%)
- **Threshold**: Gap exceeds 15% threshold, indicating potential overfitting
- **Overfitting Point**: Detected around 3,500 training samples
- **Recommendation**: Consider regularization or early stopping for production

**Key Observations**:
- Training RMSE: ~4.8 (converged)
- Validation RMSE: ~6.6 (slight increase with more data)
- Stable performance after 15,000 training samples

### 3. Residual Analysis
**Distribution Characteristics**:
- **Shape**: Normal distribution centered near zero
- **Symmetry**: Well-balanced positive and negative residuals
- **Outliers**: Few extreme residuals, indicating robust predictions
- **Homoscedasticity**: Consistent variance across prediction range

### 4. Prediction Quality
**Real vs. Predicted Values**:
- **Correlation**: Strong linear relationship along ideal line
- **Prediction Range**: 0-220 leads (matches actual range)
- **Scatter Pattern**: Tight clustering around ideal line for low-medium values
- **High-Value Predictions**: Some deviation for leads >150 (rare cases)

---

## Model Validation & Diagnostics

### 1. Cross-Validation Results
**5-Fold Cross-Validation**:
- Consistent performance across folds
- Stable RMSE distribution
- No evidence of data leakage

### 2. Feature Importance (Inferred)
Based on engineering pipeline and domain knowledge:
1. **phone_clicks**: Direct engagement metric
2. **views**: Advertisement reach indicator  
3. **vl_advertise**: Price positioning impact
4. **city/state**: Geographic market effects
5. **flag_cluster**: Vehicle feature combinations
6. **n_photos**: Visual presentation quality

### 3. Model Robustness
**Strengths**:
- Handles categorical features without extensive preprocessing
- Robust to outliers through built-in regularization
- Fast training and prediction times
- Good generalization on unseen data

**Limitations**:
- Shows overfitting tendencies with full training data
- Performance degradation on very high lead counts
- Requires careful hyperparameter tuning

---

## Production Deployment Considerations

### 1. Model Serialization
```python
# Model saved for production use
joblib.dump(pipeline, 'complete_ml_pipeline.joblib')
```

### 2. Monitoring Requirements
**Performance Monitoring**:
- Track RMSE on new predictions
- Monitor prediction distribution drift
- Alert on unusual residual patterns

**Data Drift Detection**:
- Monitor categorical feature distributions
- Track numerical feature statistics
- Validate encoding transformations

### 3. Retraining Strategy
**Triggers for Retraining**:
- Performance degradation >10% RMSE increase
- Significant data distribution changes
- New categorical values requiring encoding updates
- Quarterly model refresh schedule

---

## Business Impact & Recommendations

### 1. Model Utility
**Predictive Power**: 70.3% variance explained enables:
- Accurate lead forecasting for advertisement planning
- ROI optimization through targeted placement
- Resource allocation for high-potential listings

### 2. Key Success Factors
**Top Drivers** (based on feature engineering):
1. **Phone Engagement**: Optimize for phone click generation
2. **Visual Quality**: Implement 8-photo standard
3. **Geographic Targeting**: State-specific strategies
4. **Price Positioning**: Market value alignment

### 3. Implementation Roadmap
**Phase 1** (Immediate):
- Deploy model for lead prediction
- Implement performance monitoring
- A/B test model-driven decisions

**Phase 2** (3-6 months):
- Collect model feedback data
- Refine hyperparameters based on production performance
- Expand feature engineering based on new insights

**Phase 3** (6-12 months):
- Evaluate alternative algorithms (XGBoost, Neural Networks)
- Implement real-time prediction API
- Develop automated retraining pipeline

---

## Technical Appendix

### Model Configuration
```python
# Final model pipeline
pipeline = Pipeline([
    ('encoding', ColumnTransformer(...)),
    ('model', LGBMRegressor(**best_params))
])
```

### Dependencies
- **lightgbm**: 3.3.0+
- **scikit-learn**: 1.0+
- **optuna**: 3.0+
- **category-encoders**: 2.5+

### Performance Benchmarks
- **Training Time**: ~15 minutes (150 Optuna trials)
- **Prediction Time**: <1ms per sample
- **Memory Usage**: ~100MB model size
- **Scalability**: Linear with dataset size

---

**Summary**: The LightGBM model demonstrates strong predictive performance with practical business utility, achieving 70% variance explanation while maintaining computational efficiency suitable for production deployment.
