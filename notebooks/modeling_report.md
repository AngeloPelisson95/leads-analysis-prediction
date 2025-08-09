

### 🎯 Modeling (`modeling/`)
**Model development and evaluation:**

#### 📈 [case_parte_2.ipynb](./modeling/case_parte_2.ipynb)
- **Purpose**: Predictive modeling and evaluation
- **Content**: Model building, validation, and performance analysis

## 🚀 Quick Start Guide

### 1. **Environment Setup**
```bash
# Navigate to project root
cd /path/to/leads-analysis-prediction

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab
```

### 2. **Recommended Workflow**
Follow notebooks in sequence for complete analysis:

1. **📋 Start with Summary** → `00_analysis_summary.ipynb`
2. **📊 Data Overview** → `01_data_overview.ipynb`  
3. **🧹 Data Cleaning** → `02_data_cleaning.ipynb`
4. **🔍 Deep Analysis** → `03_exploratory_analysis.ipynb`
5. **⚙️ Feature Importance** → `04_feature_analysis.ipynb`
6. **🎯 Modeling** → `modeling/case_parte_2.ipynb`

### 3. **Key Dependencies**
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization  
- `xgboost` - Machine learning
- `shap` - Model interpretability
- `scikit-learn` - ML utilities

## 📊 Key Outputs

### Data Products:
- **Cleaned Dataset**: `data/processed/cleaned_data.csv`
- **Feature Importance Rankings**: Top predictive features identified
- **Statistical Summaries**: Comprehensive data profiling

### Business Insights:
- **Lead Generation Drivers**: Key factors influencing lead generation
- **Geographic Patterns**: State and city-level performance analysis
- **Feature Engineering**: Recommendations for model improvement
- **ROI Optimization**: Strategies for advertisement effectiveness

## 🎯 Business Value

### 📈 **Predictive Insights**
- Identify high-potential advertisements before publication
- Optimize resource allocation for lead generation
- Understand customer engagement patterns

### 🏆 **Key Findings**
- **Phone clicks** are the strongest predictor of leads
- **Photo quantity** (8 photos optimal) significantly impacts performance  
- **Geographic location** shows substantial variation in lead rates
- **Vehicle features** and **price positioning** drive engagement

### 💼 **Actionable Recommendations**
- Focus on driving phone engagement 
- Implement 8-photo standard for advertisements
- Develop state-specific marketing strategies
- Optimize price positioning relative to market value

## 🔧 Technical Notes

### **Environment Requirements**
- Python 3.8+
- Jupyter Lab/Notebook
- 4GB+ RAM recommended for full dataset analysis
- Virtual environment setup recommended

### **Data Pipeline**
```
Raw Data → Cleaning → EDA → Feature Engineering → Modeling → Insights
    ↓         ↓        ↓           ↓               ↓         ↓
 CSV files  Processed  Statistical   ML Features   Models   Business
           Data       Summaries    Importance    Results   Recommendations
```

---

**📞 Questions or Issues?** Refer to individual notebook documentation or project README for detailed guidance.
