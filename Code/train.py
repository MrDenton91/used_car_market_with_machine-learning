"""Train a used-car price model from the scraped cars.com CSVs.

The whole preprocessing + model is saved as one sklearn Pipeline, so the saved
file can price a car straight from raw values (see predict.py).

Usage:
    python Code/train.py carlist*.csv
    python Code/train.py all_data.csv --sample 200000 --out models/price_model.joblib
"""
import argparse
import json
import os

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
# the collect_*.py collectors write CSVs with a header and these extra columns
EXTRA = ['Source', 'Status']
CATEGORICAL = ['Source', 'Make', 'Model', 'Body Style', 'City', 'State', 'Color']
NUMERIC = ['Age', 'Milage']

# listings outside these ranges are data-entry errors (e.g. $100,000,000 cars)
MIN_PRICE, MAX_PRICE = 500, 500_000
MAX_MILEAGE = 500_000


def load_data(paths):
    frames = []
    for p in paths:
        with open(p) as f:
            header = f.readline()
        if header.startswith('Price,') and 'Source' in header:
            # collect_*.py output: read columns by name
            frames.append(pd.read_csv(p, low_memory=False).reindex(columns=COLUMNS + EXTRA))
        else:
            # the split*.py scraper CSVs have no header, the combined ones do; read
            # both the same way and drop any header rows that end up as data
            frames.append(pd.read_csv(p, names=COLUMNS, header=None, low_memory=False))
    data = pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS + EXTRA)
    data['Source'] = data['Source'].fillna('cars.com')
    return data[data['Price'] != 'Price']


def clean(data, scrape_year):
    data = data.copy()
    for col in ['Price', 'Year', 'Milage']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['Date Scraped'] = pd.to_datetime(data['Date Scraped'], errors='coerce')
    # a GSA auction's bid is only a sale price once the auction has closed
    data = data[(data['Source'] != 'gsa') | (data['Status'] == 'Closed')]
    # collectors leave these blank when a listing doesn't say
    data[['Body Style', 'City', 'Color']] = data[['Body Style', 'City', 'Color']].fillna('Unknown')
    # mileage may be missing (e.g. collector-car listings without a details pass);
    # the models handle that, so only the other features are required
    data = data.dropna(subset=[c for c in FEATURES if c != 'Milage'])
    for col in CATEGORICAL:
        data[col] = data[col].astype(str).str.strip()

    # a listing scraped on several days is one car; keep the day it was first added
    data = data.sort_values('Date Scraped').drop_duplicates(subset=FEATURES, keep='first')

    # age is relative to when the listing was scraped; rows from older scrapes
    # without a date fall back to --scrape-year
    year_scraped = data['Date Scraped'].dt.year.fillna(scrape_year)
    data = data[data['Price'].between(MIN_PRICE, MAX_PRICE)
                & (data['Milage'].isna() | data['Milage'].between(0, MAX_MILEAGE))
                & data['Year'].between(1900, year_scraped + 1)]
    data['Age'] = year_scraped[data.index] - data['Year']
    return data.drop(columns=['Year', 'Status']).reset_index(drop=True)


def build_models(random_state):
    one_hot = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20), CATEGORICAL),
        ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), NUMERIC),
    ])
    # City and Model have thousands of values, so the boosted model target-encodes
    # them; TargetEncoder cross-fits internally so the encoding doesn't leak the target
    target_enc = ColumnTransformer([
        ('cat', TargetEncoder(target_type='continuous'), CATEGORICAL),
        ('num', 'passthrough', NUMERIC),
    ])
    # prices are right-skewed, so fit on log(price) and report errors in dollars
    gbm = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(
            max_iter=2000, learning_rate=0.05, early_stopping=True,
            n_iter_no_change=50, random_state=random_state),
        func=np.log, inverse_func=np.exp)

    return {
        'baseline (median)': DummyRegressor(strategy='median'),
        'ridge': make_pipeline(one_hot, Ridge(alpha=1.0)),
        'gradient boosting': make_pipeline(target_enc, gbm),
    }


def evaluate(model, X, y):
    y_pred = model.predict(X)
    return {
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred)),
        'r2': float(r2_score(y, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', nargs='+', help='scraped CSV file(s)')
    parser.add_argument('--scrape-year', type=int, default=2020, help='year the data was scraped, used for car age when a row has no Date Scraped')
    parser.add_argument('--sample', type=int, help='train on a random sample of this many rows')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=10)
    parser.add_argument('--out', default='models/price_model.joblib')
    args = parser.parse_args()

    data = clean(load_data(args.data), args.scrape_year)
    if args.sample and args.sample < len(data):
        data = data.sample(args.sample, random_state=args.random_state)
    print('{:,} clean listings'.format(len(data)))

    y = data.pop('Price')
    X = data[CATEGORICAL + NUMERIC]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state)

    results = {}
    fitted = {}
    for name, model in build_models(args.random_state).items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        fitted[name] = model

    base_rmse = results['baseline (median)']['rmse']
    print('\n{:<20}{:>12}{:>12}{:>8}{:>14}'.format('model', 'RMSE', 'MAE', 'R2', 'RMSE/base'))
    for name, m in results.items():
        print('{:<20}{:>12,.0f}{:>12,.0f}{:>8.3f}{:>14.2f}'.format(
            name, m['rmse'], m['mae'], m['r2'], m['rmse'] / base_rmse))

    best = min(results, key=lambda name: results[name]['rmse'])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    joblib.dump(fitted[best], args.out)
    with open(os.path.splitext(args.out)[0] + '_metrics.json', 'w') as f:
        json.dump({'best': best, 'n_rows': len(data), 'scrape_year': args.scrape_year,
                   'results': results}, f, indent=2)
    print('\nsaved {} to {}'.format(best, args.out))


if __name__ == '__main__':
    main()
