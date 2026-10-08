"""Shared loading, cleaning and evaluation helpers for the notebooks.

Usage from a notebook in this folder:

    from car_utils import load_raw, clean, evaluate
    cars = clean(load_raw('carlist*.csv'), reference_year=2020)
"""
import datetime
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Column order of the original headerless carlist*.csv files.
LEGACY_COLUMNS = ['Price', 'Make', 'Model', 'Year', 'Body Style', 'City', 'State', 'Milage', 'Color']
# Rows missing any of these are dropped; other text columns missing a value become 'Unknown'.
REQUIRED_COLUMNS = ['Price', 'Make', 'Model', 'Year', 'Milage']
CATEGORICAL_COLUMNS = ['Body Style', 'City', 'State', 'Color']


def _read_carlist(path):
    with open(path) as f:
        first_line = f.readline()
    if first_line.startswith('Price,'):
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, low_memory=False, names=LEGACY_COLUMNS, header=None)


def load_raw(pattern='carlist*.csv'):
    """Load and concatenate every scraped carlist CSV matching ``pattern``.

    Handles both the old headerless 9-column files and the files written by
    ``scraper.py`` (with a header row and extra Trim/VIN columns).
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError('no files match {!r}'.format(pattern))
    return pd.concat([_read_carlist(p) for p in paths], ignore_index=True)


def clean(df, min_price=500, max_price=500_000, max_mileage=1_000_000, reference_year=None):
    """Drop duplicates, missing values and implausible rows, and add an ``Age`` column.

    Rows missing price, make, model, year or mileage are dropped; missing body
    style, city, state or color become 'Unknown'. Duplicates are removed by VIN
    when the data has one (the same car is often listed many times), otherwise
    by identical rows. Prices outside
    [min_price, max_price] and mileages above max_mileage are treated as
    data-entry errors. ``reference_year`` defaults to the current year; pass
    2020 to reproduce the original analysis.
    """
    df = df.copy()
    for col in ['Price', 'Year', 'Milage']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=REQUIRED_COLUMNS)
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].fillna('Unknown')

    if 'VIN' in df.columns and df['VIN'].notna().any():
        has_vin = df['VIN'].notna()
        df = pd.concat([df[has_vin].drop_duplicates(subset='VIN'), df[~has_vin].drop_duplicates()])
    else:
        df = df.drop_duplicates()

    df = df[df['Price'].between(min_price, max_price) & (df['Milage'] <= max_mileage)]
    if reference_year is None:
        reference_year = datetime.date.today().year
    df['Age'] = reference_year - df['Year'].astype(int)
    df = df[df['Age'] >= -1]  # next year's models are listed early; anything later is bad data
    return df.reset_index(drop=True)


def evaluate(model, X, y, verbose=True):
    """Score a fitted regressor against a predict-the-median baseline.

    Returns a dict with rmse, mae, r2, baseline_rmse and normalized_rmse
    (model RMSE / baseline RMSE; below 1 means better than the baseline).
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.ravel(model.predict(X))
    baseline = np.full_like(y, np.median(y))

    scores = {
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred)),
        'r2': float(r2_score(y, y_pred)),
        'baseline_rmse': float(np.sqrt(mean_squared_error(y, baseline))),
    }
    scores['normalized_rmse'] = scores['rmse'] / scores['baseline_rmse']

    if verbose:
        print(model)
        print('R2 of model: {:.2f}'.format(scores['r2']))
        print('RMSE of model: {:,.2f}'.format(scores['rmse']))
        print('MAE of model: {:,.2f}'.format(scores['mae']))
        print('Baseline RMSE (predict the median): {:,.2f}'.format(scores['baseline_rmse']))
        print('Normalized RMSE (model / baseline): {:.2f}'.format(scores['normalized_rmse']))
    return scores
