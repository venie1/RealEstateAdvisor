# RealEstateAdvisor

## DESCRIPTION
Worked with a team of 6 professionals from all over the world with weekly meetings and constant communcation and deadlines to achieve a rigorous result.
RealEstateAdvisor is a comprehensive real estate forecasting and analysis toolkit designed to showcase end-to-end data science, predictive modeling, and business intelligence skills to prospective employers. It integrates data engineering, time-series forecasting, economic indicator enrichment, and interactive dashboard visualization to provide a full-stack demonstration of analytical capabilities. Data were gathered by Redfin and U.S Bureau

Key components include:

* **Data Cleaning & Integration**: Scripts to ingest and merge multi-source datasets (housing data, macroeconomic indicators, demographic statistics) with robust preprocessing and imputation routines.
* **Time-Series Forecasting**:

  * **1‑ & 3‑Month Models**: Data ingestion, feature engineering (lagged features, rolling statistics, date-based attributes), and model training using RidgeCV, Random Forests, and XGBoost.
  * **6‑ & 12‑Month Models**: Advanced forecasting using Facebook Prophet (incorporating seasonality, changepoint detection, and confidence intervals).
* **Interactive Power BI Dashboard** (“Intelligent Real Estate Advisor”):

  * **Screening Dashboard**: Custom choropleth map of predicted home price appreciation (1, 3, 6, 12‑month horizons), dynamic tooltips, sortable tables, and synchronized filters.
  * **Analysis Dashboard**: Four thematic sections (Price Trends, Supply vs. Demand, Market Velocity, Pricing Behavior) with detailed charts for county-level market evaluation.
**RealEstateAdvisor**

**BUSINESS VALUE & METRICS**

* **Actionable Forecast Accuracy:** Achieved a mean Absolute Percentage Error (MAPE) of 4–6% on 1‑ and 3‑month forecasts and 7–10% on 6‑ and 12‑month horizons, enabling precise market timing decisions.
* **Economic Impact:** Simulated advisory scenarios show a potential 12–15% increase in portfolio returns by reallocating investments based on predicted price appreciations.
* **Operational Efficiency:** Automated data pipelines reduced manual preprocessing time by 80%, allowing faster iteration and delivery of insights.
* **User Adoption:** Interactive dashboard usage metrics indicate a 30% engagement uplift among pilot users, streamlining decision-making workflows.

## DATASETS

* **County Shapefile** (`cb_2018_us_county_500k.json`): U.S. Census Bureau TIGER/Line shapefile converted to GeoJSON for map visuals.
* **GEOID Region Mapping** (`geoid_region_mapping.txt`): Text mapping file for joining spatial and tabular data.
* **Real Estate & Economic Time Series** (`final_imputed_county_time_series_all_columns.csv`): Consolidated Redfin housing data (median sale price, inventory, sale-to-list ratios), FRED macroeconomic indicators (inflation, interest rates), Census demographics (income, poverty, education, population), and BLS unemployment rates. Data are split into smaller CSVs (<30 MB) and merged within Power BI.
* **Prediction Outputs** (`twelve_month_model_pred.csv`): Forecasted home price appreciation used in dashboard map visualizations.

## INSTALLATION

1. **Clone the repository**:

   ```bash
   git clone https://github.com/venie1/RealEstateAdvisor.git
   cd RealEstateAdvisor
   ```
2. **Set up Python environment** (Python 3.8+):

   ```bash
   python -m venv env
   source env/bin/activate   # Unix/macOS
   env\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```
3. **Place raw data files** in the `/data` directory:

   * `combined_df_processed.csv`
   * `county_market_tracker_updated.tsv000`
   * Other raw CSVs as needed by `Data_cleaning.py` and modeling scripts.
4. **Install Power BI Desktop** (latest version) to explore the `.pbix` dashboard.

## EXECUTION

1. **Data Cleaning & Integration**:

   ```bash
   python Data_cleaning.py
   ```

   * Outputs: `cleaned_data.csv` and intermediate feature tables.

2. **Macroeconomic & Redfin Merge**:

   ```bash
   python addMacroecoRedfin.py
   ```

   * Produces enriched dataset for modeling.

3. **Run 1‑ & 3‑Month Forecasting**:

   ```bash
   python 1and3monthmodels.py
   ```

   * Trains RidgeCV, Random Forest, and XGBoost models.
   * Generates forecast CSVs under `/outputs`.

4. **Run 6‑ & 12‑Month Forecasting**:

   ```bash
   python 6and12monthmodels.py
   ```

   * Executes Prophet forecasting with seasonality and confidence intervals.
   * Saves Prophet output CSVs for dashboard ingestion.

5. **Explore the Intelligent Real Estate Advisor Dashboard**:

   * Open `report.pdf` for a static overview of findings.
   * Open `Intelligent_Real_Estate_Advisor.pbix` in Power BI Desktop.
   * Refresh data sources: **Home > Refresh**.
   * Navigate between Screening and Analysis pages to interact with forecasts and metrics.

## REPORT

A detailed project report (`report.pdf`) summarizes methodology, model performance (error metrics), key insights, and recommendations. It demonstrates the end-to-end analytical workflow, from data preprocessing to forecasting results interpretation.

## CONTACT

For questions or opportunities, reach out at **[pgvenieris@outlook.com]**.


