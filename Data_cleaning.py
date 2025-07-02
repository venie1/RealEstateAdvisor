#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[2]:


def load_data(filepath, chunksize=10000):
    dfs = []
    for chunk in pd.read_csv(filepath, sep="\t", parse_dates=["period_end"], chunksize=chunksize):
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["region", "property_type", "period_end"], keep="first")
    return df


# In[3]:


def clean_columns(df):
    selected = [
        'Inflation rate', 'mortgage rate', 'fed interest rate',
        'Median Household Income', 'Persons Below Poverty Level',
        "Bachelor's Degree", 'Not in Labor Force', 'Total Population',
        'Unemployment Rate', 'property_type', 'period_end', 'region',
        'median_sale_price', 'homes_sold', 'inventory',
        'median_list_price', 'median_ppsf'
    ]
    df = df[selected]
    missing = df.isna().mean()
    return df[missing[missing <= 0.10].index]


# In[4]:


def filter_years(df, start=2020, end=2024):
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["year"] = df["period_end"].dt.year
    return df[(df["year"] >= start) & (df["year"] <= end)]


# In[5]:


def retain_sufficient_series(df, min_months=52):
    counts = df.dropna(subset=["median_sale_price"]).groupby(["region", "property_type"])["period_end"].nunique().reset_index()
    valid = counts[counts["period_end"] >= min_months][["region", "property_type"]]
    return df.merge(valid, on=["region", "property_type"])


# In[6]:


def split_train_test(df, test_start='2024-01-01'):
    df = df.sort_values(by=["region", "property_type", "period_end"])
    train = df[df["period_end"] < test_start]
    test = df[df["period_end"] >= test_start]
    return train, test


# In[7]:


def apply_grouped_rolling_capping(df, target_cols, window=12, sigma=3):
    df = df.copy()
    for col in target_cols:
        df[col] = df.groupby(["region", "property_type"])[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).apply(
                lambda s: np.clip(s.iloc[-1], s.mean() - sigma * s.std(), s.mean() + sigma * s.std())
                if len(s) > 1 else s.iloc[-1]
            )
        )
    return df


# In[8]:


def rolling_median_imputation(df, exclude_cols, window=6):
    df = df.sort_values(by=["region", "property_type", "period_end"]).copy()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    imputable_cols = [col for col in numeric_cols if col not in exclude_cols]
    for col in imputable_cols:
        df[col] = df.groupby(["region", "property_type"])[col].transform(
            lambda x: x.fillna(x.rolling(window=window, min_periods=1).median())
        )
    return df


# In[9]:


def scale_features(train_df, test_df, exclude_cols):
    numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    scaler = StandardScaler()
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
    return train_df, test_df


# In[10]:


def build_combined_df(filepath, output='combined_df_processed.csv'):
    df = load_data(filepath)
    df = clean_columns(df)
    df = filter_years(df)
    df = retain_sufficient_series(df)

    train, test = split_train_test(df)

    # all numeric columns
    numeric_cols = train.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # capping 
    train = apply_grouped_rolling_capping(train, numeric_cols)
    test = apply_grouped_rolling_capping(test, numeric_cols)

    # impute missing values using rolling median
    train = rolling_median_imputation(train, exclude_cols=[], window=6)
    test = rolling_median_imputation(test, exclude_cols=[], window=6)

    # scaling 
    train, test = scale_features(train, test, exclude_cols=["median_sale_price"])

    # combined output
    combined_df = pd.concat([train, test], axis=0).sort_values(by=["region", "property_type", "period_end"])
    combined_df.to_csv(output, index=False)
    print(f"✅ Final combined_df_processed saved to: {output}")
    return combined_df


# In[11]:


combined_df = build_combined_df("county_market_tracker.tsv000")


# In[ ]:




