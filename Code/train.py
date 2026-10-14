"""Train a used-car price model from the scraped / collected / imported CSVs.

The whole preprocessing + model is saved as one sklearn Pipeline, so the saved
file can price a car straight from raw values (see predict.py).

Usage:
    python Code/train.py carlist*.csv
    python Code/train.py data/*.csv --sample-frac 0.3 --out models/price_model.joblib
"""
import argparse
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# column order written by the split*.py scrapers; older scrapes have no 'Date Scraped' column
FEATURES = ['Price', 'Make', 'Model', 'Year', 'Body Style', 'City', 'State', 'Milage', 'Color']
COLUMNS = FEATURES + ['Date Scraped']
# the collect_*.py / import_*.py CSVs have a header and these extra columns
DETAILS = ['Trim', 'Transmission', 'Engine', 'Drive Type', 'Fuel Type', 'Title Status']
EXTRA = ['Source', 'Status', 'Listing ID'] + DETAILS
CATEGORICAL = ['Source', 'Make', 'Model', 'Body Style', 'City', 'State', 'Color'] + DETAILS
NUMERIC = ['Age', 'Sale Year', 'Milage']

# listings outside these ranges are data-entry errors (e.g. $100,000,000 cars)
MIN_PRICE, MAX_PRICE = 500, 500_000
MAX_MILEAGE = 500_000


def load_data(paths, sample_frac=1.0, sample_sources='absee', random_state=10):
    """Read the CSVs in chunks. Rows whose Source matches sample_sources (the big
    SEC loan files) are subsampled to sample_frac so the data fits in memory."""
    rng = np.random.default_rng(random_state)
    frames = []
    for p in paths:
        with open(p) as f:
            header = f.readline()
        if header.startswith('Price,') and 'Source' in header:
            reader = pd.read_csv(p, usecols=lambda c: c in COLUMNS + EXTRA, dtype=str, chunksize=1_000_000)
        else:
            # the split*.py scraper CSVs have no header, the combined ones do; read
            # both the same way and drop any header rows that end up as data
            reader = pd.read_csv(p, names=COLUMNS, header=None, dtype=str, chunksize=1_000_000)
        for chunk in reader:
            chunk = chunk.reindex(columns=COLUMNS + EXTRA)
            chunk['Source'] = chunk['Source'].fillna('cars.com')
            if sample_frac < 1.0:
                big = chunk['Source'].str.contains(sample_sources, regex=True, na=False)
                chunk = chunk[~big | (rng.random(len(chunk)) < sample_frac)]
            frames.append(chunk[chunk['Price'] != 'Price'])
    return pd.concat(frames, ignore_index=True)


def clean(data, scrape_year, include_new=False):
    data = data.copy()
    for col in ['Price', 'Year', 'Milage']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['Date Scraped'] = pd.to_datetime(data['Date Scraped'], errors='coerce')
    # a GSA auction's bid is only a sale price once the auction has closed
    data = data[(data['Source'] != 'gsa') | (data['Status'] == 'Closed')]
    if not include_new:   # loan filings also cover new cars
        data = data[data['Status'] != 'New']
    # lender names from the loan filings: 'absee:harley davidson motorcycle trust ...' -> 'absee:harley davidson'
    data['Source'] = data['Source'].str.replace(r' (motorcycle|vehicle|trust|leasing|lease)\b.*$', '', regex=True)
    # sources leave these blank when a listing doesn't say
    data[['Body Style', 'City', 'Color'] + DETAILS] = data[['Body Style', 'City', 'Color'] + DETAILS].fillna('Unknown')
    # mileage may be missing (e.g. loan filings); the models handle that, so
    # only the other features are required
    data = data.dropna(subset=[c for c in FEATURES if c != 'Milage'])
    for col in CATEGORICAL:
        data[col] = data[col].astype(str).str.strip()
    # sources spell models differently (CR-V / Crv / CRV, F-150 / F150); merge them
    data['Model'] = data['Model'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True).replace('', 'UNKNOWN')
    data['Make'] = data['Make'].str.title()

    # a listing scraped on several days is one car; keep the day it was first added
    # (rows with a Listing ID are distinct records even when their features match)
    data = data.sort_values('Date Scraped').drop_duplicates(subset=FEATURES + ['Listing ID'], keep='first')

    # age is relative to when the car was listed or sold; rows from older scrapes
    # without a date fall back to --scrape-year. The sale year is its own feature
    # because the market moved a lot between 2010 and 2026.
    data['Sale Year'] = data['Date Scraped'].dt.year.fillna(scrape_year)
    data = data[data['Price'].between(MIN_PRICE, MAX_PRICE)
                & (data['Milage'].isna() | data['Milage'].between(0, MAX_MILEAGE))
                & data['Year'].between(1900, data['Sale Year'] + 1)]
    data['Age'] = data['Sale Year'] - data['Year']
    return data.drop(columns=['Year', 'Status', 'Listing ID']).reset_index(drop=True)


def build_models(random_state, with_ridge=False):
    # City and Model have thousands of values, so the boosted model target-encodes
    # them; TargetEncoder cross-fits internally so the encoding doesn't leak the target
    target_enc = ColumnTransformer([
        ('cat', TargetEncoder(target_type='continuous'), CATEGORICAL),
        ('num', 'passthrough', NUMERIC),
    ])
    # prices are right-skewed, so fit on log(price) and report errors in dollars
    gbm = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(
            max_iter=3000, learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=50, random_state=random_state),
        func=np.log, inverse_func=np.exp)
    models = {
        'baseline (median)': DummyRegressor(strategy='median'),
        'gradient boosting': make_pipeline(target_enc, gbm),
    }
    if with_ridge:
        one_hot = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20), CATEGORICAL),
            ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), NUMERIC),
        ])
        models['ridge'] = make_pipeline(one_hot, Ridge(alpha=1.0))
    return models


def evaluate(y, y_pred):
    return {
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred)),
        'r2': float(r2_score(y, y_pred)) if len(y) > 1 else float('nan'),
        'n': int(len(y)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', nargs='+', help='CSV file(s)')
    parser.add_argument('--scrape-year', type=int, default=2020, help='year the data was scraped, used when a row has no date')
    parser.add_argument('--sample', type=int, help='train on a random sample of this many rows')
    parser.add_argument('--sample-frac', type=float, default=1.0, help='keep this fraction of rows from --sample-sources while reading')
    parser.add_argument('--sample-sources', default='absee', help='regex of sources to subsample (default: the SEC loan files)')
    parser.add_argument('--include-new', action='store_true', help='also train on new-car records (SEC loan filings)')
    parser.add_argument('--with-ridge', action='store_true', help='also fit a ridge regression (slow on large data)')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=10)
    parser.add_argument('--out', default='models/price_model.joblib')
    args = parser.parse_args()

    data = clean(load_data(args.data, args.sample_frac, args.sample_sources, args.random_state),
                 args.scrape_year, args.include_new)
    if args.sample and args.sample < len(data):
        data = data.sample(args.sample, random_state=args.random_state)
    print('{:,} clean listings'.format(len(data)), flush=True)

    y = data.pop('Price')
    X = data[CATEGORICAL + NUMERIC]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state)

    results, fitted, predictions = {}, {}, {}
    for name, model in build_models(args.random_state, args.with_ridge).items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        results[name] = evaluate(y_test, predictions[name])
        fitted[name] = model

    base_rmse = results['baseline (median)']['rmse']
    print('\n{:<20}{:>12}{:>12}{:>8}{:>14}'.format('model', 'RMSE', 'MAE', 'R2', 'RMSE/base'))
    for name, m in results.items():
        print('{:<20}{:>12,.0f}{:>12,.0f}{:>8.3f}{:>14.2f}'.format(
            name, m['rmse'], m['mae'], m['r2'], m['rmse'] / base_rmse))

    best = min(results, key=lambda name: results[name]['rmse'])
    by_source = {}
    print('\n{} by source:'.format(best))
    print('{:<28}{:>10}{:>12}{:>12}{:>8}'.format('source', 'rows', 'RMSE', 'MAE', 'R2'))
    for source, idx in X_test.groupby('Source').indices.items():
        m = evaluate(y_test.iloc[idx], predictions[best][idx])
        by_source[source] = m
        print('{:<28}{:>10,}{:>12,.0f}{:>12,.0f}{:>8.3f}'.format(source, m['n'], m['rmse'], m['mae'], m['r2']))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    joblib.dump(fitted[best], args.out)
    with open(os.path.splitext(args.out)[0] + '_metrics.json', 'w') as f:
        json.dump({'best': best, 'n_rows': len(data), 'scrape_year': args.scrape_year,
                   'features': CATEGORICAL + NUMERIC, 'results': results, 'by_source': by_source}, f, indent=2)
    print('\nsaved {} to {}'.format(best, args.out))


if __name__ == '__main__':
    main()
