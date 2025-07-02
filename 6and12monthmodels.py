#!/usr/bin/env python
# coding: utf-8
#  Prophet Forecasting Pipeline
# -Implements a Prophet-based forecasting system with:
# - 2020–2022 train, 2023 validate
# - Hyperparameter tuning
# - Forecasting for 6- and 12-month ahead horizons (June and December 2024)
# - YoY % growth calculations
# - COVID-19 and Ukraine War holidays
# - Clean output saved as prophet_predictions_with_yoy.csv

# In[1]:


import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import time
import os


# In[2]:


start_time = time.time()

def print_progress(step, start_time):
    elapsed = (time.time() - start_time) / 60
    print(f"[{elapsed:.2f} min] {step}")


# In[3]:


print_progress("Loading data...", start_time)
df_filtered = pd.read_csv('combined_df_processed.csv', parse_dates=['period_end'])
df_filtered = df_filtered[df_filtered['period_end'] >= '2020-01-01'].copy()
df_filtered = df_filtered[df_filtered['property_type'] != 'All Residential'].copy()
df_filtered = df_filtered[['period_end', 'region', 'property_type', 'median_sale_price']]
df_filtered.set_index(['period_end', 'region', 'property_type'], inplace=True)
print_progress("Data loaded", start_time)


# In[4]:


covid_holidays = pd.DataFrame({
    'holiday': 'COVID-19',
    'ds': pd.to_datetime(['2020-03-01']),
    'lower_window': -7,
    'upper_window': 60
})
ukraine_war = pd.DataFrame({
    'holiday': 'Ukraine War',
    'ds': pd.to_datetime(['2022-02-24']),
    'lower_window': -7,
    'upper_window': 60
})
holidays = pd.concat([covid_holidays, ukraine_war])
print_progress("Holidays defined", start_time)


# In[5]:


forecast_dates = pd.to_datetime(['2024-06-30', '2024-12-31'])


# In[6]:


def prophet_cv_tune(prophet_df, param_grid):
    prophet_df = prophet_df[prophet_df['ds'] <= '2023-12-31']
    train = prophet_df[prophet_df['ds'] < '2023-01-01']
    val = prophet_df[(prophet_df['ds'] >= '2023-01-01') & (prophet_df['ds'] <= '2023-12-31')]
    if len(train) < 24 or len(val) < 6:
        return None, None
    best_score = np.inf
    best_model = None
    best_params = None
    for params in param_grid:
        try:
            model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode=params['seasonality_mode'],
                holidays=holidays
            )
            model.fit(train)
            future = val[['ds']].copy()
            forecast = model.predict(future)
            merged = val.merge(forecast[['ds', 'yhat']], on='ds')
            score = mean_absolute_percentage_error(merged['y'], merged['yhat'])
            if score < best_score:
                best_score = score
                best_model = model
                best_params = params
        except:
            continue
    return best_model, best_params


# In[7]:


def fit_group(region, property_type, group, forecast_dates, param_grid):
    prophet_df = group[['period_end', 'median_sale_price']].rename(columns={'period_end': 'ds', 'median_sale_price': 'y'})
    if len(prophet_df) < 36:
        return None
    model, best_params = prophet_cv_tune(prophet_df.copy(), param_grid)
    if model is None:
        return None
    try:
        future = pd.DataFrame({'ds': forecast_dates})
        forecast = model.predict(future)
        forecast['region'] = region
        forecast['property_type'] = property_type
        forecast['y'] = np.nan
        for i, date in enumerate(forecast_dates):
            actual = prophet_df[prophet_df['ds'] == date]['y']
            if not actual.empty:
                forecast.loc[i, 'y'] = actual.values[0]
        return forecast[['ds', 'yhat', 'y', 'region', 'property_type']]
    except Exception as e:
        print(f"Model failed for {region}-{property_type}: {e}")
        return None


# In[8]:


def train_and_forecast(df, forecast_dates, param_grid, batch_size=500):
    groups = list(df.groupby(['region', 'property_type']))
    total_groups = len(groups)
    forecasts = []
    for i in range(0, total_groups, batch_size):
        chunk = groups[i:i + batch_size]
        chunk_forecasts = Parallel(n_jobs=-1)(delayed(fit_group)(region, property_type, group, forecast_dates, param_grid)
                                              for (region, property_type), group in chunk)
        chunk_df = pd.concat([f for f in chunk_forecasts if f is not None], ignore_index=True)
        chunk_df.to_csv(f'forecast_chunk_{i}.csv', index=False)
        forecasts.append(chunk_df)
        print_progress(f"Processed batch {i} to {min(i + batch_size - 1, total_groups - 1)}", start_time)
    forecast_df = pd.concat(forecasts, ignore_index=True)
    for i in range(0, total_groups, batch_size):
        os.remove(f'forecast_chunk_{i}.csv')
    print_progress("Forecasting completed", start_time)
    return forecast_df


# In[9]:


def calculate_yoy_all_horizons(forecast_df, df_filtered, target='median_sale_price'):
    print_progress("Starting YoY calculation...", start_time)
    yoy_results = []
    horizons = [
        ('2024-06-30', '2023-06-30'),
        ('2024-12-31', '2023-12-31')
    ]
    for date, prev_year_date in horizons:
        date = pd.to_datetime(date)
        prev_year_date = pd.to_datetime(prev_year_date)
        current = forecast_df[forecast_df['ds'] == date].copy()
        previous = df_filtered.loc[prev_year_date].reset_index()
        previous = previous[previous['region'].isin(current['region']) & 
                           previous['property_type'].isin(current['property_type'])].rename(columns={target: f'{target}_prev'})
        merged = current.merge(previous[['region', 'property_type', f'{target}_prev']], on=['region', 'property_type'])
        merged = merged.dropna(subset=[f'{target}_prev'])
        merged['yoy_actual'] = (merged['y'] - merged[f'{target}_prev']) / merged[f'{target}_prev'] * 100
        merged['yoy_predicted'] = (merged['yhat'] - merged[f'{target}_prev']) / merged[f'{target}_prev'] * 100
        merged['horizon'] = date
        yoy_results.append(merged)
        print_progress(f"YoY calculated for {date.strftime('%B %Y')}", start_time)
    yoy_df = pd.concat(yoy_results, ignore_index=True)
    print_progress("YoY calculation completed", start_time)
    return yoy_df


# In[10]:


param_grid = [
    {'changepoint_prior_scale': 0.05, 'seasonality_mode': 'additive'},
    {'changepoint_prior_scale': 0.1, 'seasonality_mode': 'additive'},
    {'changepoint_prior_scale': 0.07, 'seasonality_mode': 'multiplicative'}, 
    {'changepoint_prior_scale': 0.03, 'seasonality_mode': 'additive'},
    {'changepoint_prior_scale': 0.07, 'seasonality_mode': 'additive'},
    {'changepoint_prior_scale': 0.03, 'seasonality_mode': 'multiplicative'}
]

forecast_df = train_and_forecast(df_filtered.reset_index(), forecast_dates, param_grid)
yoy_df = calculate_yoy_all_horizons(forecast_df, df_filtered)

print_progress("Saving final output...", start_time)
merged_output = pd.merge(
    forecast_df,
    yoy_df[['region', 'property_type', 'ds', 'yoy_actual', 'yoy_predicted']],
    on=['region', 'property_type', 'ds'],
    how='left'
)

merged_output.rename(columns={
    'ds': 'period_end',
    'y': 'median_sale_price',
    'yhat': 'median_sale_price_predicted'
}, inplace=True)

final_columns = ['region', 'property_type', 'period_end', 'median_sale_price',
                 'median_sale_price_predicted', 'yoy_actual', 'yoy_predicted']
merged_output = merged_output[final_columns]
merged_output.to_csv('prophet_predictions_with_yoy.csv', index=False)
print_progress("Final results saved to prophet_predictions_with_yoy.csv", start_time)





