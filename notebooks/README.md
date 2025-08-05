# Notebooks Directory

This directory contains Jupyter notebooks for the Lead Generation Vehicle Listings analysis, organized by purpose:

## Structure
- `exploratory/`: Notebooks for data exploration, lead analysis, and visualization
- `modeling/`: Notebooks for predictive model development, training, and evaluation

## Guidelines
- Use descriptive names with prefixes (e.g., `01_data_exploration.ipynb`)
- Include a brief description at the top of each notebook
- Clear the output before committing to version control
- Use markdown cells to document your thought process

## Current Notebooks
- `exploratory/data_eda.ipynb`: Initial vehicle data exploration and lead analysis
- `exploratory/case_desafio_bonus.ipynb`: Bonus challenge - advanced lead optimization analysis
- `modeling/case_parte_2.ipynb`: Predictive model development for lead generation

## Findings


**🚗 VEHICLE-RELATED FEATURES**

✅ `cd_vehicle_brand, cd_model_vehicle, cd_version_vehicle`

* High cardinality: 75 brands, 656 models, 4,149 versions.
* Heavy tail: Many values with extremely low frequency (e.g., 0.000021).
* Top-heavy: A small subset dominates (e.g., top model 2848 has 3% of data).

**Implications:**
* Consider target encoding or frequency binning.
* Group rare brands/models into "others".

✅ `year_model`

* Range: modern cars dominate (2013–2018 are most common).
* Some suspicious values (e.g., -1, 1952–1964).

**Implications:**
* Clean or filter out invalid/very old years.
* Bucket into “age groups” (e.g., 0–5 years, 6–10 years, etc.).

✅ `fuel_type, transmission_type, n_doors`
* fuel_type: Mostly flex (gasolina e alcool – 72%), diesel is only 5.5%.
* transmission_type: Mostly manual (55%) and automatico (43%) — others are rare.
* n_doors: Almost all cars have 4 doors (88%).

**Implications**
* One-hot encoding works well here (few dominant categories).
* Consider grouping rare options.

✅ `Mileage & Price (km_vehicle, vl_advertise, vl_market)`

* All are numeric with high cardinality and heavy tails.
* Should be treated with binning (e.g., price ranges, km ranges) or scaling + log transform.
* Extreme values like 405000000 for vl_advertise may be outliers or errors.

**Implications:**
* Clean and cap outliers, log-transform prices/km if skewed.

📸 **AD ATTRIBUTES**
✅ `n_photos`

* Most ads have 8 photos (~54%), few have 0–4.

* This is highly imbalanced.

**Implications:**
* Could be a strong feature: more photos → more trust.
* Consider bucketing: 0, 1-4, 5-7, 8+.

✅ `views, phone_clicks`

* Wide value ranges (e.g., views up to 93k, phone_clicks up to 457).
* Likely power-law distribution: long tail.

**Implications:**
* Treat as numeric but possibly log-transform.
* These could be strong proxy features for lead generation.

🏷️ **BINARY FLAGS**
✅ `Flags (flg_single_owner, flg_license, flg_tax_paid, etc.)`

* Many features with skewed distributions:
    * flg_all_dealership_services → only 15% have this
    * flg_factory_warranty → only 8.5%
    * flg_pcd, flg_armored → extremely rare

**Implications:**

* Keep most flags as-is (binary), but beware of:
* Extremely rare features: may add noise unless they are very predictive.
* Use SHAP or feature importance to select the useful ones.

🌎 **LOCATION**
✅ `city, state, zip_2dig`

* state has manageable cardinality (27 + N/A).
* city and zip_2dig have high cardinality and many rare values.

**Implications**:

Encode state as categorical or one-hot.

* For city/zip:
    * Consider grouping rare ones into "other".
    * Or target encoding / clustering ZIPs by socioeconomic factors if available.
    ---

🧍 **USER / CLIENT FEATURES**

✅ `cd_client`, `cd_advertise`
* Extremely high cardinality (~48k ads, ~16k clients).
* All proportions are very small and uniform.

**Implications:**
* Likely not useful directly unless you compute aggregates per client (e.g., avg. leads/client).
* Or treat as grouping variables.

✅ `cd_type_individual` (person vs business?)
* Skewed: ~85% of one type (likely individuals).

**Implications:**
* Could be predictive — include as binary.

✅ `priority`
* 3 categories: dominated by level 3 (~77%).
* Could reflect importance/urgency of ad.

**Implications:**
* Keep as categorical; might be predictive of lead generation.

---

🎯 **TARGET VARIABLE**

✅ `leads`
* Long-tail distribution, most common values: 1, 0, 4, 8.
* 21% of ads have zero leads.
* The findings until now we have the target proportion, showing that 78% of the advertisers have some lead.

**Implications:**
* You can:
    * Predict binary (lead > 0)
    * Predict count via regression
    * Build a scoring model to prioritize ads

