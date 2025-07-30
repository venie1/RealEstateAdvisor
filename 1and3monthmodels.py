

# Standard and third-party library imports
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import multiprocessing
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import RidgeCV, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# File paths for input data
imputed_path = 'combined_df_processed.csv'
county_market_path = 'county_market_tracker_updated.tsv000'

# Load and preprocess the main imputed dataset
imputed_df = pd.read_csv(imputed_path)
imputed_df.rename(columns={'period_end': 'period_begin'}, inplace=True)
imputed_df['period_begin'] = pd.to_datetime(imputed_df['period_begin'])
regions = imputed_df['region'].unique()
earliest_date = imputed_df['period_begin'].min()

# Load and filter the county market data
county_market_df = pd.read_csv(county_market_path, sep='\t')
county_market_df['period_begin'] = pd.to_datetime(county_market_df['period_begin'])
filtered_county_market_df = county_market_df[
    (county_market_df['region'].isin(regions)) &
    (county_market_df['period_begin'] >= earliest_date)
]

# Select relevant columns for modeling
columns_to_keep = [
    'median_sale_price', 'inventory', 'homes_sold', 'median_dom', 'avg_sale_to_list',
    'pending_sales', 'new_listings', 'sold_above_list', 'median_ppsf',
    'B19013_001E', 'B17001_002E', 'B15003_022E', 'B23025_005E', 'B01003_001E',
    'mortgage rate', 'fed interest rate', 'period_begin', 'region', 'property_type'
]
filtered_county_market_df = filtered_county_market_df[columns_to_keep]
filtered_county_market_df['month_period'] = filtered_county_market_df['period_begin'].dt.to_period('M')

def create_complete_monthly_df(df, group_cols):
    """
    Generate a DataFrame with complete monthly periods for each group.
    Missing months are filled with forward-filled values for each group.
    """
    monthly_df = df.groupby(group_cols + ['month_period']).mean(numeric_only=True).reset_index()
    full_month_range = pd.period_range(start=filtered_county_market_df['period_begin'].min(),
                                       end=filtered_county_market_df['period_begin'].max(),
                                       freq='M')
    complete_list = []
    for group_val in monthly_df[group_cols[0]].unique():
        group_df = monthly_df[monthly_df[group_cols[0]] == group_val].set_index('month_period')
        group_df = group_df.reindex(full_month_range)
        group_df[group_cols[0]] = group_val
        for col in group_cols[1:]:
            group_df[col] = group_df[col].ffill()
        group_df = group_df.reset_index().rename(columns={'index': 'month_period'})
        complete_list.append(group_df)
    complete_monthly_df = pd.concat(complete_list, ignore_index=True)
    complete_monthly_df['period_begin'] = complete_monthly_df['month_period'].dt.to_timestamp()
    other_cols = [col for col in complete_monthly_df.columns if col not in group_cols + ['period_begin', 'month_period']]
    final_cols = group_cols + ['period_begin'] + other_cols
    complete_monthly_df = complete_monthly_df[final_cols]
    return complete_monthly_df

def impute_series(df, metrics_cols, training_medians=None):
    """
    Impute missing values in the given columns using forward fill, backward fill, and optionally training medians.
    """
    df = df.sort_values('period_begin').copy()
    df[metrics_cols] = df[metrics_cols].fillna(method='ffill')
    df[metrics_cols] = df[metrics_cols].fillna(method='bfill')
    if training_medians is not None:
        for col in metrics_cols:
            df[col] = df[col].fillna(training_medians.get(col, df[col].median() if not df[col].empty else 0))
    else:
        for col in metrics_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
    return df

def calculate_yoy(df):
    """
    Calculate year-over-year (YoY) percentage change in median sale price for each region.
    """
    df = df.sort_values(['region', 'period_begin']).copy()
    df['prev_year_price'] = df.groupby('region')['median_sale_price'].shift(12)
    df['yoy'] = (df['median_sale_price'] - df['prev_year_price']) / df['prev_year_price']
    df['yoy'] = np.where((df['prev_year_price'] == 0) | (df['prev_year_price'].isna()), np.nan, df['yoy'])
    df['yoy'] = (df['yoy'] * 100).round(2)
    df.drop(columns=['prev_year_price'], inplace=True)
    return df

# Prepare a complete monthly DataFrame for all regions
total_monthly_df = create_complete_monthly_df(filtered_county_market_df, group_cols=['region'])

# Split data into training and testing sets for each region
train_list, test_list = [], []
for county in total_monthly_df['region'].unique():
    county_data = total_monthly_df[total_monthly_df['region'] == county].sort_values('period_begin')
    split_idx = int(0.8 * len(county_data))
    train_list.append(county_data.iloc[:split_idx])
    test_list.append(county_data.iloc[split_idx:])
train_df_full = pd.concat(train_list)
test_df_full = pd.concat(test_list)
metrics_columns_full = [col for col in total_monthly_df.columns if col not in ['region', 'period_begin']]

# Impute missing values for training data
imputed_train_list = []
for county in train_df_full['region'].unique():
    county_train = train_df_full[train_df_full['region'] == county].copy()
    training_medians = county_train[metrics_columns_full].median()
    county_train_imputed = impute_series(county_train, metrics_columns_full, training_medians)
    imputed_train_list.append(county_train_imputed)
imputed_train_df_full = pd.concat(imputed_train_list, ignore_index=True)

# Impute missing values for test data using training medians or global medians
imputed_test_list = []
for county in test_df_full['region'].unique():
    county_test = test_df_full[test_df_full['region'] == county].copy().sort_values('period_begin')
    county_train = train_df_full[train_df_full['region'] == county].copy().sort_values('period_begin')
    if not county_train.empty:
        training_medians = county_train[metrics_columns_full].median()
        last_train_row = county_train.iloc[-1][metrics_columns_full]
        seed_row = last_train_row.to_frame().T
        seed_row['period_begin'] = county_test['period_begin'].min() - pd.Timedelta(days=1)
        seed_row['region'] = county
        county_test_seeded = pd.concat([seed_row, county_test], ignore_index=True)
        county_test_seeded = county_test_seeded.sort_values('period_begin')
        county_test_seeded[metrics_columns_full] = county_test_seeded[metrics_columns_full].fillna(method='ffill')
        county_test_seeded[metrics_columns_full] = county_test_seeded[metrics_columns_full].fillna(method='bfill')
        county_test_imputed = county_test_seeded.iloc[1:]
        for col in metrics_columns_full:
            county_test_imputed[col] = county_test_imputed[col].fillna(training_medians[col])
    else:
        global_train_medians = train_df_full[metrics_columns_full].median()
        county_test_imputed = county_test.copy()
        county_test_imputed[metrics_columns_full] = county_test_imputed[metrics_columns_full].fillna(global_train_medians)
    imputed_test_list.append(county_test_imputed)
imputed_test_df_full = pd.concat(imputed_test_list, ignore_index=True)

# Combine imputed training and test data for final analysis
final_imputed_df_full = pd.concat([imputed_train_df_full, imputed_test_df_full], ignore_index=True)
final_imputed_df_full.sort_values(['region', 'period_begin'], inplace=True)
final_imputed_df_full = calculate_yoy(final_imputed_df_full)

# Process data separately for each property type
property_types = filtered_county_market_df['property_type'].unique()
final_dfs = {}
for ptype in property_types:
    df_ptype = filtered_county_market_df[filtered_county_market_df['property_type'] == ptype].copy()
    monthly_df = create_complete_monthly_df(df_ptype, group_cols=['region', 'property_type'])
    monthly_df.drop(columns=['property_type'], inplace=True)
    train_list, test_list = [], []
    for county in monthly_df['region'].unique():
        county_data = monthly_df[monthly_df['region'] == county].sort_values('period_begin')
        split_idx = int(0.8 * len(county_data))
        train_list.append(county_data.iloc[:split_idx])
        test_list.append(county_data.iloc[split_idx:])
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    metrics_columns = [col for col in monthly_df.columns if col not in ['region', 'period_begin']]
    imputed_train_list = []
    for county in train_df['region'].unique():
        county_train = train_df[train_df['region'] == county].copy()
        training_medians = county_train[metrics_columns].median()
        county_train_imputed = impute_series(county_train, metrics_columns, training_medians)
        imputed_train_list.append(county_train_imputed)
    imputed_train_df = pd.concat(imputed_train_list, ignore_index=True)
    imputed_test_list = []
    for county in test_df['region'].unique():
        county_test = test_df[test_df['region'] == county].copy().sort_values('period_begin')
        county_train = train_df[train_df['region'] == county].copy().sort_values('period_begin')
        if not county_train.empty:
            training_medians = county_train[metrics_columns].median()
            last_train_row = county_train.iloc[-1][metrics_columns]
            seed_row = last_train_row.to_frame().T
            seed_row['period_begin'] = county_test['period_begin'].min() - pd.Timedelta(days=1)
            seed_row['region'] = county
            county_test_seeded = pd.concat([seed_row, county_test], ignore_index=True)
            county_test_seeded = county_test_seeded.sort_values('period_begin')
            county_test_seeded[metrics_columns] = county_test_seeded[metrics_columns].fillna(method='ffill')
            county_test_seeded[metrics_columns] = county_test_seeded[metrics_columns].fillna(method='bfill')
            county_test_imputed = county_test_seeded.iloc[1:]
            for col in metrics_columns:
                county_test_imputed[col] = county_test_imputed[col].fillna(training_medians[col])
        else:
            global_train_medians = train_df[metrics_columns].median()
            county_test_imputed = county_test.copy()
            county_test_imputed[metrics_columns] = county_test_imputed[metrics_columns].fillna(global_train_medians)
        imputed_test_list.append(county_test_imputed)
    imputed_test_df = pd.concat(imputed_test_list, ignore_index=True)
    final_imputed_df = pd.concat([imputed_train_df, imputed_test_df], ignore_index=True)
    final_imputed_df.sort_values(['region', 'period_begin'], inplace=True)
    final_imputed_df = calculate_yoy(final_imputed_df)
    final_dfs[ptype] = final_imputed_df.copy()

print("\nProcessing complete.")


# In[2]:


def create_complete_monthly_df(df, group_cols):
    if df.empty or pd.isna(df['period_begin']).all():
        print("Warning: Empty DataFrame or all NaT values in period_begin")
        empty_df = pd.DataFrame(columns=group_cols + ['period_begin'])
        return empty_df
    
    df = df.dropna(subset=['period_begin'])
    if df.empty:
        print("Warning: DataFrame is empty after dropping NaT values")
        empty_df = pd.DataFrame(columns=group_cols + ['period_begin'])
        return empty_df
    if 'month_period' not in df.columns:
        df['month_period'] = df['period_begin'].dt.to_period('M')
    monthly_df = df.groupby(group_cols + ['month_period']).mean(numeric_only=True).reset_index()
    if monthly_df.empty:
        print("Warning: No data after grouping")
        empty_df = pd.DataFrame(columns=group_cols + ['period_begin'])
        return empty_df
    try:
        full_month_range = pd.period_range(start=df['period_begin'].min(), 
                                          end=df['period_begin'].max(),
                                          freq='M')
    except Exception as e:
        empty_df = pd.DataFrame(columns=group_cols + ['period_begin'])
        return empty_df
    complete_list = []
    for group_val in monthly_df[group_cols[0]].unique():
        group_df = monthly_df[monthly_df[group_cols[0]] == group_val].set_index('month_period')
        group_df = group_df.reindex(full_month_range)
        group_df[group_cols[0]] = group_val  # reintroduce region
        for col in group_cols[1:]:
            group_df[col] = group_df[col].ffill()
        group_df = group_df.reset_index().rename(columns={'index': 'month_period'})
        complete_list.append(group_df)
    if not complete_list:
        empty_df = pd.DataFrame(columns=group_cols + ['period_begin'])
        return empty_df
        
    complete_monthly_df = pd.concat(complete_list, ignore_index=True)
    complete_monthly_df['period_begin'] = complete_monthly_df['month_period'].dt.to_timestamp()
    other_cols = [col for col in complete_monthly_df.columns if col not in group_cols + ['period_begin', 'month_period']]
    final_cols = group_cols + ['period_begin'] + other_cols
    complete_monthly_df = complete_monthly_df[final_cols]
    return complete_monthly_df

def impute_series(df, metrics_cols, training_medians=None):
    if df.empty or not metrics_cols:
        return df
        
    df = df.sort_values('period_begin').copy()
    df[metrics_cols] = df[metrics_cols].fillna(method='ffill')
    df[metrics_cols] = df[metrics_cols].fillna(method='bfill')
    if training_medians is not None:
        for col in metrics_cols:
            if col in df.columns:  # Make sure column exists
                df[col] = df[col].fillna(training_medians.get(col, df[col].median() if not df[col].empty else 0))
    else:
        for col in metrics_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if not df[col].empty else 0)
    return df

def calculate_yoy(df):
    if df.empty or 'combined_region' not in df.columns or 'median_sale_price' not in df.columns:
        print("Warning: Cannot calculate YoY - missing required columns")
        if 'combined_region' in df.columns:  # Add yoy column even if we can't calculate it
            df['yoy'] = np.nan
        return df
        
    try:
        df = df.sort_values(['combined_region', 'period_begin']).copy()
        df['prev_year_price'] = df.groupby('combined_region')['median_sale_price'].shift(12)
        df['yoy'] = (df['median_sale_price'] - df['prev_year_price']) / df['prev_year_price']
        df['yoy'] = np.where((df['prev_year_price'] == 0) | (df['prev_year_price'].isna()), np.nan, df['yoy'])
        df['yoy'] = (df['yoy'] * 100).round(2)
        df.drop(columns=['prev_year_price'], inplace=True)
    except Exception as e:
        print(f"Error calculating YoY: {str(e)}")
        df['yoy'] = np.nan
    return df

def process_imputation(df, group_col='combined_region', output_filename_prefix='processed'):
    if df.empty:
        print(f"Warning: Empty DataFrame for {output_filename_prefix}")
        return df

    if 'period_begin' in df.columns:
        df['period_begin'] = pd.to_datetime(df['period_begin'], errors='coerce')
    else:
        print(f"Warning: No period_begin column in {output_filename_prefix}")
        return df
    df = df.dropna(subset=['period_begin'])
    
    if df.empty:
        print(f"Warning: No valid dates in DataFrame for {output_filename_prefix}")
        return df
    
    try:
        monthly_df = create_complete_monthly_df(df, group_cols=[group_col])
        if monthly_df.empty:
            print(f"Warning: Empty monthly DataFrame for {output_filename_prefix}")
            return df
        
        metrics_cols = [col for col in monthly_df.columns if col not in [group_col, 'period_begin']]
        if not metrics_cols:
            print(f"Warning: No metrics columns found for {output_filename_prefix}")
            return monthly_df
        
        train_list, test_list = [], []
        for group in monthly_df[group_col].unique():
            group_df = monthly_df[monthly_df[group_col] == group].sort_values('period_begin')
            if len(group_df) > 0:
                split_idx = max(1, int(0.8 * len(group_df)))  # Ensure at least 1 row in train
                train_list.append(group_df.iloc[:split_idx])
                if split_idx < len(group_df):  # Only add to test if there's data left
                    test_list.append(group_df.iloc[split_idx:])
        if not train_list:
            print(f"Warning: No training data for {output_filename_prefix}")
            return monthly_df
        
        train_df = pd.concat(train_list, ignore_index=True)
        
        imputed_train_list = []
        for group in train_df[group_col].unique():
            group_train = train_df[train_df[group_col] == group].copy()
            if not group_train.empty:
                training_medians = group_train[metrics_cols].median(numeric_only=True)
                group_train_imputed = impute_series(group_train, metrics_cols, training_medians)
                imputed_train_list.append(group_train_imputed)
        
        if not imputed_train_list:
            print(f"Warning: No imputed training data for {output_filename_prefix}")
            return train_df
        
        imputed_train_df = pd.concat(imputed_train_list, ignore_index=True)
        if test_list:
            test_df = pd.concat(test_list, ignore_index=True)
            imputed_test_list = []
            
            for group in test_df[group_col].unique():
                group_test = test_df[test_df[group_col] == group].copy().sort_values('period_begin')
                if group_test.empty:
                    continue
                    
                group_train = train_df[train_df[group_col] == group].copy().sort_values('period_begin')
                
                if not group_train.empty:
                    training_medians = group_train[metrics_cols].median(numeric_only=True)
                    try:
                        last_train_row = group_train.iloc[-1][metrics_cols]
                        seed_row = last_train_row.to_frame().T
                        seed_row['period_begin'] = group_test['period_begin'].min() - pd.Timedelta(days=1)
                        seed_row[group_col] = group
                        group_test_seeded = pd.concat([seed_row, group_test], ignore_index=True)
                        group_test_seeded = group_test_seeded.sort_values('period_begin')
                        for col in metrics_cols:
                            if col in group_test_seeded.columns:
                                group_test_seeded[col] = group_test_seeded[col].fillna(method='ffill')
                                group_test_seeded[col] = group_test_seeded[col].fillna(method='bfill')
                        group_test_imputed = group_test_seeded.iloc[1:]
                        for col in metrics_cols:
                            if col in group_test_imputed.columns:
                                group_test_imputed[col] = group_test_imputed[col].fillna(training_medians.get(col, group_test_imputed[col].median() if not group_test_imputed[col].empty else 0))
                    except Exception as e:
                        print(f"Error seeding test data for group {group}: {str(e)}")
                        group_test_imputed = group_test.copy()
                        global_train_medians = train_df[metrics_cols].median(numeric_only=True)
                        for col in metrics_cols:
                            if col in group_test_imputed.columns:
                                group_test_imputed[col] = group_test_imputed[col].fillna(global_train_medians.get(col, group_test_imputed[col].median() if not group_test_imputed[col].empty else 0))
                else:
                    global_train_medians = train_df[metrics_cols].median(numeric_only=True)
                    group_test_imputed = group_test.copy()
                    for col in metrics_cols:
                        if col in group_test_imputed.columns:
                            group_test_imputed[col] = group_test_imputed[col].fillna(global_train_medians.get(col, group_test_imputed[col].median() if not group_test_imputed[col].empty else 0))
                
                imputed_test_list.append(group_test_imputed)
            
            if imputed_test_list:
                imputed_test_df = pd.concat(imputed_test_list, ignore_index=True)
                final_imputed_df = pd.concat([imputed_train_df, imputed_test_df], ignore_index=True)
            else:
                print(f"Warning: No imputed test data for {output_filename_prefix}")
                final_imputed_df = imputed_train_df
        else:
            print(f"Warning: No test data for {output_filename_prefix}, using train data only")
            final_imputed_df = imputed_train_df
        final_imputed_df.sort_values([group_col, 'period_begin'], inplace=True)
        final_imputed_df = calculate_yoy(final_imputed_df)
        
        return final_imputed_df
    
    except Exception as e:
        print(f"Error in process_imputation for {output_filename_prefix}: {str(e)}")
        return df



condo_df = final_dfs.get("Condo/Co-op", pd.DataFrame())
all_residential_df = final_dfs.get("All Residential", pd.DataFrame())
single_family_df = final_dfs.get("Single Family Residential", pd.DataFrame())
multi_family_df = final_dfs.get("Multi-Family (2-4 Unit)", pd.DataFrame())
townhouse_df = final_dfs.get("Townhouse", pd.DataFrame())

# Make backup copies before renaming columns
if not condo_df.empty and 'region' in condo_df.columns:
    condo_df = condo_df.copy()
    condo_df.rename(columns={'region': 'combined_region'}, inplace=True)

if not all_residential_df.empty and 'region' in all_residential_df.columns:
    all_residential_df = all_residential_df.copy()
    all_residential_df.rename(columns={'region': 'combined_region'}, inplace=True)

if not single_family_df.empty and 'region' in single_family_df.columns:
    single_family_df = single_family_df.copy()
    single_family_df.rename(columns={'region': 'combined_region'}, inplace=True)

if not multi_family_df.empty and 'region' in multi_family_df.columns:
    multi_family_df = multi_family_df.copy()
    multi_family_df.rename(columns={'region': 'combined_region'}, inplace=True)

if not townhouse_df.empty and 'region' in townhouse_df.columns:
    townhouse_df = townhouse_df.copy()
    townhouse_df.rename(columns={'region': 'combined_region'}, inplace=True)

# Normalize region names
for df in [condo_df, all_residential_df, single_family_df, multi_family_df, townhouse_df]:
    if not df.empty and 'combined_region' in df.columns:
        df['combined_region'] = df['combined_region'].astype(str).str.lower().str.strip()

# Set up property mapping
property_mapping = {
    'condo': ['condo/co-op'],
    'multi_family': ['multi-family (2-4 unit)'],
    'townhouse': ['townhouse'],
    'all_residential': ['all residential'],
    'single_family': ['single family', 'single-family']
}

# Get valid regions for each property type
combined_df = pd.read_csv("combined_df_processed.csv")
if not combined_df.empty:
    if 'property_type' in combined_df.columns:
        combined_df['property_type'] = combined_df['property_type'].astype(str).str.lower().str.strip()
    if 'region' in combined_df.columns:
        combined_df['region'] = combined_df['region'].astype(str).str.lower().str.strip()

    valid_regions = {}
    for prop, variants in property_mapping.items():
        if 'property_type' in combined_df.columns and 'region' in combined_df.columns:
            regions = combined_df.loc[combined_df['property_type'].str.lower().isin([v.lower() for v in variants]), 'region'].unique().tolist()
            valid_regions[prop] = [r for r in regions if pd.notnull(r)]
        else:
            valid_regions[prop] = []

# Process datasets
datasets = {
    "condo": condo_df,
    "all_residential": all_residential_df,
    "single_family": single_family_df,
    "multi_family": multi_family_df,
    "townhouse": townhouse_df
}

for name, df in datasets.items():
    
    # Skip if DataFrame is empty
    if df.empty:
        print(f"Warning: Empty DataFrame for {name}, skipping")
        continue
    
    # Filter for valid regions if they exist
    if name in valid_regions and valid_regions[name]:
        df = df[df['combined_region'].isin([r.lower() for r in valid_regions[name]])]
    
    # Skip if DataFrame is empty after filtering
    if df.empty:
        print(f"Warning: No valid regions found for {name}, skipping")
        continue
    
    try:
        processed_df = process_imputation(df, group_col='combined_region', output_filename_prefix=f"final_imputed_{name}")
        datasets[name] = processed_df
    except Exception as e:
        print(f"Error processing {name} dataset: {str(e)}")
        # Keep the original DataFrame
        datasets[name] = df

# Update the original variables with processed data
condo_df = datasets["condo"]
all_residential_df = datasets["all_residential"]
single_family_df = datasets["single_family"]
multi_family_df = datasets["multi_family"]
townhouse_df = datasets["townhouse"]

print("\nAll datasets have been imputed, filtered and updated.")


# In[3]:


# Define a mapping to use the search names (first element of each list from your mapping)
search_name_mapping = {
    'condo': property_mapping['condo'][0],
    'multi_family': property_mapping['multi_family'][0],
    'townhouse': property_mapping['townhouse'][0],
    'all_residential': property_mapping['all_residential'][0],
    'single_family': property_mapping['single_family'][0]
}
combined_list = []
for prop, df in datasets.items():
    df = df.copy()  # Avoid modifying the original DataFrame directly
    # Assign the search name from the mapping dictionary to the property_type column
    df['property_type'] = search_name_mapping.get(prop, prop)
    combined_list.append(df)

overall_df = pd.concat(combined_list, ignore_index=True)
overall_df.rename(columns={'combined_region': 'region', 'period_begin': 'period_end'}, inplace=True)
print("\nOverall combined dataset created")


# In[4]:


import warnings
import pandas as pd

# At the very top of your script, BEFORE any imports that might use pandas
pd.options.mode.chained_assignment = None  # Default is 'warn'

# Also suppress both specific warning types
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Then your other imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from joblib import Parallel, delayed  # For parallel processing
from tqdm import tqdm              # For progress bar
from tqdm_joblib import tqdm_joblib  # To integrate tqdm with joblib
def county_level_time_series_prediction(df):
    df['period_begin'] = pd.to_datetime(df['period_begin'])
    df = df.sort_values(['combined_region', 'period_begin'])
    counties = df['combined_region'].unique()
    total_counties = len(counties)
    print(f"Predicting median sale prices for {total_counties} counties")
    
    # Process counties in parallel with a progress bar
    with tqdm_joblib(tqdm(desc="Processing counties", total=total_counties)) as progress_bar:
        processed = Parallel(n_jobs=-1)(
            delayed(process_county)(county, df) for county in counties
        )
    
    results = {}
    county_predictions = {}
    for county, county_results, county_forecast in processed:
        if county_results is not None:
            results[county] = county_results
        if county_forecast is not None:
            county_predictions[county] = county_forecast
    summary_df = create_summary_report(results, county_predictions)
    return results, county_predictions, summary_df


def county_level_time_series_prediction_subset(df, n_counties=100):
    df['period_begin'] = pd.to_datetime(df['period_begin'])
    df = df.sort_values(['combined_region', 'period_begin'])
    counties = df['combined_region'].unique()[:n_counties]
    total_counties = len(counties)
    print(f"Predicting median sale prices for {total_counties} counties (first {n_counties})")
    
    # Process counties in parallel with a progress bar
    with tqdm_joblib(tqdm(desc="Processing counties", total=total_counties)) as progress_bar:
        processed = Parallel(n_jobs=-1)(
            delayed(process_county)(county, df) for county in counties
        )
    
    results = {}
    county_predictions = {}
    for county, county_results, county_forecast in processed:
        if county_results is not None:
            results[county] = county_results
        if county_forecast is not None:
            county_predictions[county] = county_forecast
    summary_df = create_summary_report(results, county_predictions)
    return results, county_predictions, summary_df


def process_county(county, df):
    """
    Helper function to process a single county.
    """
    # Add these lines at the beginning of the function
    import pandas as pd
    import warnings
    pd.options.mode.chained_assignment = None
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    county_data = df[df['combined_region'] == county].copy()
    if len(county_data) < 12:
        return county, None, None
    try:
        county_results, county_forecast = predict_for_county(county_data, verbose=False)
        return county, county_results, county_forecast
    except Exception as e:
        return county, {'error': str(e)}, None


def predict_for_county(county_data, verbose=False):
    train_data, test_data = train_test_split_ts(county_data, verbose=verbose)
    county_results = {}
    # Run only Linear, RandomForest, and XGBoost
    linear_results = train_linear_model(train_data, test_data, verbose=verbose)
    county_results['Linear'] = linear_results
    rf_results = train_random_forest(train_data, test_data, verbose=verbose)
    county_results['RandomForest'] = rf_results
    xgb_results = train_xgboost(train_data, test_data, verbose=verbose)
    county_results['XGBoost'] = xgb_results
    best_model = find_best_model(county_results, verbose=verbose)
    next_period = pd.Timestamp(test_data['period_begin'].max()) + pd.DateOffset(months=1)
    forecast = generate_forecast(train_data, best_model, next_period)
    county_forecast = {
        'county': county_data['combined_region'].iloc[0],
        'next_period': next_period,
        'forecast': forecast,
        'best_model': best_model,
        'previous_price': county_data['median_sale_price'].iloc[-1],
        'percent_change': (forecast - county_data['median_sale_price'].iloc[-1]) / county_data['median_sale_price'].iloc[-1] * 100
    }
    return county_results, county_forecast


def train_test_split_ts(data, test_size=0.2, verbose=False):
    n = len(data)
    split_idx = int(n * (1 - test_size))
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    if verbose:
        print(f"  Training data: {train.shape} (from {train['period_begin'].min()} to {train['period_begin'].max()})")
        print(f"  Testing data: {test.shape} (from {test['period_begin'].min()} to {test['period_begin'].max()})")
    return train, test


def prepare_features(train_data, test_data=None):
    X_train = train_data.copy()
    X_train['month'] = X_train['period_begin'].dt.month
    X_train['year'] = X_train['period_begin'].dt.year
    if len(X_train) > 3:
        X_train['median_sale_price_lag1'] = X_train['median_sale_price'].shift(1)
        X_train['median_sale_price_lag3'] = X_train['median_sale_price'].shift(3)
        X_train['median_sale_price_ma3'] = X_train['median_sale_price'].rolling(window=3).mean()
        X_train['price_change_1m'] = X_train['median_sale_price'].pct_change(1)
        X_train['price_change_3m'] = X_train['median_sale_price'].pct_change(3)
    X_train = X_train.dropna()
    feature_cols = ['month', 'year']
    potential_features = ['median_sale_price_lag1', 'median_sale_price_lag3', 'median_sale_price_ma3',
                          'price_change_1m', 'price_change_3m', 'median_list_price', 'inventory',
                          'homes_sold', 'median_dom', 'avg_sale_to_list', 'pending_sales',
                          'new_listings', 'sold_above_list', 'median_ppsf',
                          'B19013_001E', 'B17001_002E', 'B15003_022E', 'B23025_005E',
                          'B01003_001E', 'mortgage rate', 'fed interest rate']
    for col in potential_features:
        if col in X_train.columns:
            feature_cols.append(col)
    
    if test_data is not None:
        X_test = test_data.copy()
        X_test['month'] = X_test['period_begin'].dt.month
        X_test['year'] = X_test['period_begin'].dt.year
        for col in train_data.columns:
            if col != 'period_begin' and col in X_test.columns:
                X_test.loc[X_test.index[0], col] = train_data[col].iloc[-1]
        if 'median_sale_price_lag1' in feature_cols:
            last_train_price = X_train['median_sale_price'].iloc[-1]
            X_test['median_sale_price_lag1'] = X_test['median_sale_price'].shift(1)
            if len(X_test) > 0:
                X_test['median_sale_price_lag1'].iloc[0] = last_train_price
        if 'median_sale_price_lag3' in feature_cols:
            if len(X_train) >= 3:
                last_three_prices = X_train['median_sale_price'].iloc[-3:].values
                while len(last_three_prices) < 3:
                    last_three_prices = np.insert(last_three_prices, 0, last_three_prices[0])
                X_test['median_sale_price_lag3'] = X_test['median_sale_price'].shift(3)
                for i in range(min(3, len(X_test))):
                    if i == 0:
                        X_test['median_sale_price_lag3'].iloc[i] = last_three_prices[0]
                    elif i == 1:
                        X_test['median_sale_price_lag3'].iloc[i] = last_three_prices[1]
                    elif i == 2:
                        X_test['median_sale_price_lag3'].iloc[i] = last_three_prices[2]
            else:
                if 'median_sale_price_lag3' in feature_cols:
                    feature_cols.remove('median_sale_price_lag3')
        if 'median_sale_price_ma3' in feature_cols:
            X_test['median_sale_price_ma3'] = X_test['median_sale_price'].rolling(window=3).mean()
            for i in range(min(3, len(X_test))):
                n_train_vals_needed = 3 - i
                if len(X_train) >= n_train_vals_needed:
                    last_train_vals = X_train['median_sale_price'].iloc[-n_train_vals_needed:].values
                    test_vals_to_use = X_test['median_sale_price'].iloc[:i].values
                    all_vals = np.concatenate((last_train_vals, test_vals_to_use))
                    X_test['median_sale_price_ma3'].iloc[i] = all_vals.mean()
                else:
                    all_train = X_train['median_sale_price'].values
                    test_vals_to_use = X_test['median_sale_price'].iloc[:i].values
                    all_vals = np.concatenate((all_train, test_vals_to_use))
                    X_test['median_sale_price_ma3'].iloc[i] = all_vals.mean()
        if 'price_change_1m' in feature_cols:
            X_test['price_change_1m'] = X_test['median_sale_price'].pct_change(1)
            if len(X_test) > 0:
                last_train_price = X_train['median_sale_price'].iloc[-1]
                first_test_price = X_test['median_sale_price'].iloc[0]
                X_test['price_change_1m'].iloc[0] = (first_test_price - last_train_price) / last_train_price
        if 'price_change_3m' in feature_cols:
            X_test['price_change_3m'] = X_test['median_sale_price'].pct_change(3)
            for i in range(min(3, len(X_test))):
                n_train_vals_needed = 3 - i
                if len(X_train) >= n_train_vals_needed:
                    reference_price = X_train['median_sale_price'].iloc[-n_train_vals_needed]
                    current_price = X_test['median_sale_price'].iloc[i]
                    X_test['price_change_3m'].iloc[i] = (current_price - reference_price) / reference_price
        for col in feature_cols:
            if col not in X_test.columns:
                raise ValueError(f"Test data is missing required feature: {col}")
        return X_train, X_test, feature_cols
    return X_train, feature_cols

def evaluate_model(y_true, y_pred, model_name, verbose=False):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    if verbose:
        print(f"  {model_name} Performance:")
        print(f"    MAE: ${mae:.2f}")
        print(f"    RMSE: ${rmse:.2f}")
        print(f"    RÂ²: {r2:.4f}")
        print(f"    MAPE: {mape:.2f}%")
    return {'model': model_name, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


def train_linear_model(train_data, test_data, verbose=False):
    try:
        X_train, X_test, feature_cols = prepare_features(train_data, test_data)
        y_train = X_train['median_sale_price']
        y_test = X_test['median_sale_price']
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0]))
        ])
        pipeline.fit(X_train[feature_cols], y_train)
        y_pred = pipeline.predict(X_test[feature_cols])
        results = evaluate_model(y_test, y_pred, "Ridge Regression", verbose)
        results.update({
            'model_obj': pipeline,
            'predictions': y_pred,
            'feature_cols': feature_cols,
            'y_test': y_test,
            'dates': X_test['period_begin'],
            'train_dates': X_train['period_begin'],
            'train_values': X_train['median_sale_price']
        })
        return results
    except Exception as e:
        if verbose:
            print(f"  Ridge Regression failed: {str(e)}")
        return {'model': "Ridge Regression", 'mae': float('inf'), 'rmse': float('inf'),
                'r2': float('-inf'), 'mape': float('inf'), 'error': str(e)}


def train_random_forest(train_data, test_data, verbose=False):
    try:
        X_train, X_test, feature_cols = prepare_features(train_data, test_data)
        y_train = X_train['median_sale_price']
        y_test = X_test['median_sale_price']
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
        tscv = TimeSeriesSplit(n_splits=3)
        rf = RandomForestRegressor(random_state=42)
        grid_rf = GridSearchCV(rf, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_rf.fit(X_train[feature_cols], y_train)
        best_rf = grid_rf.best_estimator_
        y_pred = best_rf.predict(X_test[feature_cols])
        results = evaluate_model(y_test, y_pred, "Random Forest", verbose)
        results.update({
            'model_obj': best_rf,
            'predictions': y_pred,
            'feature_cols': feature_cols,
            'y_test': y_test,
            'dates': X_test['period_begin'],
            'train_dates': X_train['period_begin'],
            'train_values': X_train['median_sale_price']
        })
        if verbose:
            print("  Best RF parameters:", grid_rf.best_params_)
        return results
    except Exception as e:
        if verbose:
            print(f"  Random Forest failed: {str(e)}")
        return {'model': "Random Forest", 'mae': float('inf'), 'rmse': float('inf'),
                'r2': float('-inf'), 'mape': float('inf'), 'error': str(e)}


def train_xgboost(train_data, test_data, verbose=False):
    try:
        X_train, X_test, feature_cols = prepare_features(train_data, test_data)
        y_train = X_train['median_sale_price']
        y_test = X_test['median_sale_price']
        param_grid = {'n_estimators': [50, 100, 200],
                      'learning_rate': [0.01, 0.1, 0.2],
                      'max_depth': [3, 5, 7]}
        tscv = TimeSeriesSplit(n_splits=3)
        xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
        grid_xgb = GridSearchCV(xgb, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_xgb.fit(X_train[feature_cols], y_train)
        best_xgb = grid_xgb.best_estimator_
        y_pred = best_xgb.predict(X_test[feature_cols])
        results = evaluate_model(y_test, y_pred, "XGBoost", verbose)
        results.update({
            'model_obj': best_xgb,
            'predictions': y_pred,
            'feature_cols': feature_cols,
            'y_test': y_test,
            'dates': X_test['period_begin'],
            'train_dates': X_train['period_begin'],
            'train_values': X_train['median_sale_price']
        })
        if verbose:
            print("  Best XGBoost parameters:", grid_xgb.best_params_)
        return results
    except Exception as e:
        if verbose:
            print(f"  XGBoost failed: {str(e)}")
        return {'model': "XGBoost", 'mae': float('inf'), 'rmse': float('inf'),
                'r2': float('-inf'), 'mape': float('inf'), 'error': str(e)}


def find_best_model(models_results, verbose=False):
    best_model = None
    best_r2 = float('-inf')
    for model_name, results in models_results.items():
        if 'error' in results or results['r2'] is None:
            continue
        if results['r2'] > best_r2:
            best_r2 = results['r2']
            best_model = model_name
    if best_model is None:
        best_model = 'Linear'  # Fallback to Linear if all models fail
    if verbose:
        print(f"  Best model: {best_model} (R²: {best_r2:.4f})")
    return best_model

def train_model(train_data, model_type):
    X_train, feature_cols = prepare_features(train_data)
    y_train = X_train['median_sale_price']
    if model_type == 'Linear':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0]))
        ])
    elif model_type == 'RandomForest':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'XGBoost':
        model = XGBRegressor(random_state=42, objective='reg:squarederror')
    else:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0]))
        ])
    model.fit(X_train[feature_cols], y_train)
    return model, feature_cols


def prepare_features_forecast(train_data, forecast_date):
    last_obs = train_data.iloc[-1]
    forecast_features = {}
    forecast_features['month'] = forecast_date.month
    forecast_features['year'] = forecast_date.year
    forecast_features['median_sale_price_lag1'] = last_obs['median_sale_price']
    if len(train_data) >= 3:
        forecast_features['median_sale_price_lag3'] = train_data['median_sale_price'].iloc[-3]
    else:
        forecast_features['median_sale_price_lag3'] = last_obs['median_sale_price']
    if len(train_data) >= 3:
        forecast_features['median_sale_price_ma3'] = train_data['median_sale_price'].iloc[-3:].mean()
    else:
        forecast_features['median_sale_price_ma3'] = last_obs['median_sale_price']
    if len(train_data) >= 2:
        second_last = train_data['median_sale_price'].iloc[-2]
        forecast_features['price_change_1m'] = (last_obs['median_sale_price'] - second_last) / second_last
    else:
        forecast_features['price_change_1m'] = 0
    if len(train_data) >= 3:
        third_last = train_data['median_sale_price'].iloc[-3]
        forecast_features['price_change_3m'] = (last_obs['median_sale_price'] - third_last) / third_last
    else:
        forecast_features['price_change_3m'] = 0
    for col in ['median_list_price', 'inventory', 'homes_sold', 'median_dom', 
                'avg_sale_to_list', 'pending_sales', 'new_listings', 'sold_above_list', 
                'median_ppsf', 'B19013_001E', 'B17001_002E', 'B15003_022E', 'B23025_005E', 
                'B01003_001E', 'mortgage rate', 'fed interest rate']:
        if col in train_data.columns:
            forecast_features[col] = train_data[col].iloc[-1]
    df_forecast = pd.DataFrame([forecast_features])
    df_forecast['period_begin'] = forecast_date
    return df_forecast


def generate_forecast(train_data, model_type, forecast_date):
    model, feature_cols = train_model(train_data, model_type)
    X_forecast = prepare_features_forecast(train_data, forecast_date)
    return model.predict(X_forecast[feature_cols])[0]

def compile_best_predictions(model_results, county_predictions):
    all_predictions = []
    for county, forecast_info in county_predictions.items():
        if 'best_model' not in forecast_info:
            continue
        best_model = forecast_info['best_model']
        county_results = model_results.get(county, {})
        model_data = county_results.get(best_model, {})
        if not all(key in model_data for key in ['dates', 'y_test', 'predictions']):
            continue
        county_df = pd.DataFrame({
            'combined_region': county,
            'period_begin': model_data['dates'],
            'actual_median_sale_price': model_data['y_test'],
            'best_model_prediction': model_data['predictions'],
            'best_model': best_model
        })
        all_predictions.append(county_df)
    if not all_predictions:
        return pd.DataFrame()
    return pd.concat(all_predictions, ignore_index=True)


def create_summary_report(results, forecasts):
    summary = []
    for county, forecast in forecasts.items():
        try:
            best_model = forecast.get('best_model')
            model_results = results.get(county, {}).get(best_model, {})
            summary.append({
                'County': forecast['county'],
                'Forecast': forecast['forecast'],
                'Previous Price': forecast['previous_price'],
                'Percent Change': forecast['percent_change'],
                'Best Model': best_model,
                'MAE': model_results.get('mae', np.nan),
                'RMSE': model_results.get('rmse', np.nan),
                'R2': model_results.get('r2', np.nan),
                'MAPE': model_results.get('mape', np.nan)
            })
        except KeyError as ke:
            print(f"Missing key in forecast for county {county}: {ke}")
    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('Percent Change', ascending=False)
    print("\n=== Overall Model Performance ===")
    if 'R2' in summary_df.columns:
        print(f"Average R²: {summary_df['R2'].mean():.4f}")
    if 'MAE' in summary_df.columns:
        print(f"Median MAE: {summary_df['MAE'].median():.2f}")
    if 'RMSE' in summary_df.columns:
        print(f"Median RMSE: {summary_df['RMSE'].median():.2f}")
    if 'MAPE' in summary_df.columns:
        print(f"Median MAPE: {summary_df['MAPE'].median():.2f}%")
    return summary_df

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

# For the FutureWarning
    warnings.filterwarnings('ignore', category=FutureWarning)
    model_results1, predictions1, summary_df1 = county_level_time_series_prediction(townhouse_df)
    bestdf1 = compile_best_predictions(model_results1, predictions1)
    model_results2, predictions2, summary_df2 = county_level_time_series_prediction(condo_df)
    bestdf2 = compile_best_predictions(model_results2, predictions2)
    model_results3, predictions3, summary_df3 = county_level_time_series_prediction(multi_family_df)
    bestdf3 = compile_best_predictions(model_results3, predictions3)
    model_results4, predictions4, summary_df4 = county_level_time_series_prediction(single_family_df)
    bestdf4 = compile_best_predictions(model_results4, predictions4)
    model_results5, predictions5, summary_df5 = county_level_time_series_prediction(all_residential_df)
    bestdf5 = compile_best_predictions(model_results5, predictions5)


# In[5]:


# =============================================================================
# Post-processing Additions
# =============================================================================
def enhance_predictions(predictions_dict, model_results_dict, final_df):
    pd.options.mode.chained_assignment = None
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    for county, forecast in predictions_dict.items():
        county_data = final_df[final_df['combined_region'] == county]
        forecast_date = forecast['next_period']
        try:
            prev_year_date = forecast_date - pd.DateOffset(years=1)
            prev_year_value = county_data[county_data['period_begin'] == prev_year_date]['median_sale_price'].values[0]
            forecast['yoy_growth_pct'] = ((forecast['forecast'] - prev_year_value) / prev_year_value * 100)
        except (IndexError, KeyError):
            forecast['yoy_growth_pct'] = np.nan
        best_model = forecast['best_model']
        try:
            forecast['county_r2'] = model_results_dict[county][best_model]['r2']
        except KeyError:
            forecast['county_r2'] = np.nan
    
    return predictions_dict

def create_enhanced_summary(predictions_dict):
    return pd.DataFrame([
        {
            'county': v['county'],
            'forecast_date': v['next_period'],
            'forecast_price': v['forecast'],
            'yoy_growth_pct': v.get('yoy_growth_pct', np.nan),
            'county_r2': v.get('county_r2', np.nan),
            'best_model': v['best_model'],
            'mom_change_pct': v['percent_change']
        } for v in predictions_dict.values()
    ])

# Apply to all your predictions and save to CSV
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    pd.options.mode.chained_assignment = None
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    enhanced_summary1 = create_enhanced_summary(predictions1)
    enhanced_summary2 = create_enhanced_summary(predictions2)
    enhanced_summary3 = create_enhanced_summary(predictions3)
    enhanced_summary4 = create_enhanced_summary(predictions4)
    enhanced_summary5 = create_enhanced_summary(predictions5)



# In[6]:


import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
dataframe_mappings = [
    (enhanced_summary1, "townhouse"),
    (enhanced_summary2, "condo"),
    (enhanced_summary3, "multi-family"),
    (enhanced_summary4, "single-family"),
    (enhanced_summary5, "all residential")
]
dfs = []
for df, prop_type in dataframe_mappings:
    df_copy = df.copy()
    
    df_copy['property_type'] = prop_type
    
    df_copy = df_copy.rename(columns={
        'forecast_date': 'date',
        'forecasted_price': 'forecasted_price',
        'yoy_growth_pct': 'forecast_yoy'
    })
    
    df_copy['forecast_yoy'] = df_copy['forecast_yoy'] / 100
    df_copy['forecast_yoy'] = df_copy['forecast_yoy'].round(2)
    df_copy = df_copy[['county', 'date', 'forecasted_price', 'forecast_yoy', 'property_type']]
    
    dfs.append(df_copy)
final_df = pd.concat(dfs, ignore_index=True)




# In[7]:


import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import multiprocessing
import joblib

# Sklearn and XGBoost imports:
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ---------------------- Timer Utility ----------------------
timing_stats = {}

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        timing_stats[self.name] = elapsed

def print_timing_summary():
    print("\n===== TIMING SUMMARY =====")
    total_time = sum(timing_stats.values())
    for name, duration in sorted(timing_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (duration / total_time) * 100 if total_time > 0 else 0

# ---------------------- Data Processing ----------------------
def process_group(group_data, group_cols=['region', 'property_type']):
    group = group_data.sort_values('period_end').copy()
    lag_cols = ['median_sale_price', 'median_ppsf', 'inventory', 'homes_sold', 'median_list_price']
    lag_periods = [1, 3, 6, 12]
    for col in lag_cols:
        if col in group.columns:
            for lag in lag_periods:
                group[f'{col}_lag{lag}'] = group[col].shift(lag)
    
    for window in [3, 6, 12]:
        for col in ['median_sale_price', 'inventory']:
            if col in group.columns:
                group[f'{col}_rolling_mean_{window}'] = group[col].rolling(window=window, min_periods=1).mean()
                group[f'{col}_rolling_std_{window}'] = group[col].rolling(window=window, min_periods=1).std()
    
    for period in [3, 6]:
        if 'median_sale_price' in group.columns:
            group[f'price_momentum_{period}m'] = group['median_sale_price'].pct_change(periods=period)
    
    for col in ['inventory', 'homes_sold']:
        if col in group.columns:
            group[f'{col}_mom_change'] = group[col].pct_change()
    
    if 'homes_sold' in group.columns and 'inventory' in group.columns:
        group['sales_velocity'] = group['homes_sold'] / group['inventory'].replace(0, np.nan)
    
    return group

def create_lagged_features(df, group_cols=['region', 'property_type']):
    groups = df.groupby(group_cols)
    results = Parallel(n_jobs=max(1, multiprocessing.cpu_count()-1))(
        delayed(process_group)(group, group_cols) for _, group in groups
    )
    return pd.concat(results, axis=0)

def add_time_features(df):
    df = df.copy()
    df['year'] = df['period_end'].dt.year
    df['month'] = df['period_end'].dt.month
    df['quarter'] = df['period_end'].dt.quarter
    df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                          'Spring' if x in [3, 4, 5] else
                                          'Summer' if x in [6, 7, 8] else
                                          'Fall')
    return df

def process_target_group(group, horizon):
    group = group.sort_values('period_end').copy()
    group[f'target_price_{horizon}m'] = group['median_sale_price'].shift(-horizon)
    group[f'target_date_{horizon}m'] = group['period_end'] + pd.DateOffset(months=horizon)
    return group

def create_target_variable(df, horizon=3):
    groups = df.groupby(['region', 'property_type'])
    results = Parallel(n_jobs=max(1, multiprocessing.cpu_count()-1))(
        delayed(process_target_group)(group, horizon) for _, group in groups
    )
    result = pd.concat(results, axis=0)
    return result

# ---------------------- Feature Preparation ----------------------
def prepare_features(train_data, target_col):
    skip_cols = [col for col in train_data.columns if 'target' in col]
    base_numerical = [
        "median_ppsf", "median_list_price", "Median Household Income",
        "inventory", "homes_sold", "mortgage rate", "Unemployment Rate",
        "Local Unemployment", "Sales-to-Inventory"
    ]
    lag_features = [col for col in train_data.columns if ('lag' in col or 'rolling' in col or 'momentum' in col or 'change' in col)]
    selected_numerical = [col for col in base_numerical + lag_features + ['sales_velocity']
                          if col in train_data.columns and col not in skip_cols]
    selected_categorical = ['property_type', 'month', 'quarter', 'season', 'region']
    selected_categorical = [col for col in selected_categorical if col in train_data.columns]
    selected_features = selected_numerical + selected_categorical
    X_train = train_data[selected_features].copy()
    y_train = train_data[target_col]
    
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, selected_numerical),
            ('cat', categorical_transformer, selected_categorical)
        ],
        remainder='drop'
    )
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    return preprocessor, X_train_processed, y_train, selected_features

# ---------------------- Model Training ----------------------
def train_models(X_train, y_train):
    model_results = {}
    
    models = {
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,
                                                      subsample=0.8, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5,
                                random_state=42, n_jobs=max(1, multiprocessing.cpu_count()-1),
                                tree_method='hist')
    }
    for name, model in models.items():
        with Timer(f"{name} Training"):
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
            print(f"{name} Training MAPE: {mape:.2f}%")
            model_results[name] = {
                'model_obj': model,
                'train_mape': mape
            }
    
    return model_results

def generate_future_predictions(df_clean, preprocessor, feature_cols, models, horizon=3):
    recent_data = df_clean.sort_values('period_end').groupby(['region', 'property_type']).last().reset_index()
    recent_data['forecast_date'] = recent_data['period_end'] + pd.DateOffset(months=horizon)
    X_new = recent_data[feature_cols].copy()
    X_new_processed = preprocessor.transform(X_new)
    predictions = {}
    for model_name, model_info in models.items():
        model = model_info['model_obj']
        preds = model.predict(X_new_processed)
        predictions[f'{model_name}_prediction'] = preds
    results_df = recent_data[['region', 'property_type', 'period_end', 'forecast_date', 'median_sale_price']].copy()
    for model_name, preds in predictions.items():
        results_df[model_name] = preds
    model_cols = [col for col in results_df.columns if '_prediction' in col]
    results_df['Ensemble_prediction'] = results_df[model_cols].mean(axis=1)
    for model_name in list(models.keys()) + ['Ensemble']:
        results_df[f'{model_name}_yoy'] = (results_df[f'{model_name}_prediction'] / results_df['median_sale_price']) - 1
    
    # Format and save the results
    results_df = results_df.rename(columns={'region': 'county'})
    results_df = results_df.sort_values(['county', 'property_type'])
    simple_df = results_df[['county', 'property_type', 'period_end', 'forecast_date', 
                            'median_sale_price', 'Ensemble_prediction', 'Ensemble_yoy']].copy()
    simple_df = simple_df.rename(columns={
        'period_end': 'current_date',
        'forecast_date': 'prediction_date',
        'Ensemble_prediction': 'forecasted_price',
        'Ensemble_yoy': 'forecast_yoy'
    })
    
    return results_df, simple_df

# ---------------------- Main Pipeline ----------------------
def run_housing_price_prediction_pipeline(horizon=3):
    warnings.filterwarnings('ignore')
    sns.set_style("whitegrid")
    np.random.seed(42)
    with Timer("Data Processing"):
        df = overall_df.copy()  # Use the global variable 'overall_df'
        df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
        df = df.dropna(subset=['period_end'])
        df['period_end'] = df['period_end'] + pd.offsets.MonthEnd(0)
        df_with_features = create_lagged_features(df)
        df_with_features = add_time_features(df_with_features)
        df_with_target = create_target_variable(df_with_features, horizon=horizon)
        lag_columns = [col for col in df_with_target.columns if 'lag' in col]
        df_clean = df_with_target.dropna(subset=lag_columns, thresh=3)
    
    with Timer("Model Training"):
        target_col = f'target_price_{horizon}m'
        train_data = df_clean.dropna(subset=[target_col])
        preprocessor, X_train, y_train, feature_cols = prepare_features(train_data, target_col)
        model_results = train_models(X_train, y_train)
    
    with Timer("Generate Future Predictions"):
        results_df, simple_df = generate_future_predictions(
            df_clean=df_clean,
            preprocessor=preprocessor,
            feature_cols=feature_cols,
            models=model_results,
            horizon=horizon
        )
    return results_df, simple_df

# ---------------------- Execute the Pipeline ----------------------
if __name__ == "__main__":
    results_df, simple_df = run_housing_price_prediction_pipeline(horizon=3)
    print("\n3-Month Future Predictions made")


# In[8]:


import pandas as pd
import numpy as np
historical_file = 'combined_df_processed.csv'
def normalize_county(county):
    parts = county.split(',')
    if len(parts) == 2:
        county_name = parts[0].strip().title()
        state_abbr = parts[1].strip().upper()
        return f"{county_name}, {state_abbr}"
    else:
        return county.title()
property_mapping = {
    'single-family': 'Single Family Residential',
    'single family': 'Single Family Residential',
    'residential condo': 'condo/co-op',
    'condo': 'condo/co-op',
    'condo/co-op': 'condo/co-op',
    'multi-family': 'multi-family',
    'multi-family (2-4 unit)': 'multi-family',
    'multi family (2-4 unit)': 'multi-family',
    'all residential': 'all residential',
    'townhouse': 'townhouse'
}
historical_df = pd.read_csv(historical_file)
historical_df['region'] = historical_df['region'].apply(normalize_county)
historical_df['property_type'] = historical_df['property_type'].str.lower().map(lambda x: property_mapping.get(x, x))
historical_df['period_end'] = pd.to_datetime(historical_df['period_end'])
df_1m = final_df
df_1m = df_1m.rename(columns={
    'date': 'forecast_date',
    'forecasted_price': 'predicted_price',
    'forecast_yoy': 'predicted_yoy'
})
df_1m['horizon'] = 1
df_1m['county'] = df_1m['county'].apply(normalize_county)
df_1m['property_type'] = df_1m['property_type'].str.lower().map(lambda x: property_mapping.get(x, x))
df_1m['forecast_date'] = pd.to_datetime(df_1m['forecast_date'])
df_3m = simple_df
df_3m = df_3m.rename(columns={
    'prediction_date': 'forecast_date',
    'forecasted_price': 'predicted_price',
    'forecast_yoy': 'predicted_yoy'
})
df_3m['horizon'] = 3
df_3m['county'] = df_3m['county'].apply(normalize_county)
df_3m['property_type'] = df_3m['property_type'].str.lower().map(lambda x: property_mapping.get(x, x))
df_3m['forecast_date'] = pd.to_datetime(df_3m['forecast_date'])
df_combined = pd.concat([df_1m, df_3m], ignore_index=True)
def calculate_yoy(row, hist_df):
    one_year_ago = row['forecast_date'] - pd.DateOffset(years=1)
    historical_rows = hist_df[(hist_df['region'] == row['county']) & 
                             (hist_df['property_type'] == row['property_type'])]
    
    if historical_rows.empty:
        return np.nan
    historical_rows['date_diff'] = abs((historical_rows['period_end'] - one_year_ago).dt.days)
    
    if len(historical_rows) > 0:
        closest_row = historical_rows.loc[historical_rows['date_diff'].idxmin()]
        last_year_price = closest_row['median_sale_price']
        
        if pd.notna(last_year_price) and last_year_price > 0:
            # Calculate YOY as decimal (current/previous - 1)
            yoy = (row['predicted_price'] / last_year_price) - 1
            return yoy
    
    return np.nan
for idx, row in df_combined.iterrows():
    if pd.isna(row['predicted_yoy']):
        df_combined.at[idx, 'predicted_yoy'] = calculate_yoy(row, historical_df)

df_combined = df_combined[['county', 'forecast_date', 'predicted_price', 'predicted_yoy', 'property_type', 'horizon']]
df_combined['forecast_date'] = df_combined['forecast_date'].dt.strftime('%Y-%m-%d')
df_combined = df_combined.sort_values(by=['county', 'property_type']).reset_index(drop=True)
print(df_combined)
output_file = '1and3monthpredictions.csv'
df_combined.to_csv(output_file, index=False)
print(f"Combined data saved to {output_file}")

