"""Check the price range (10th- and 90th-percentile models) on the held-out test split.

Reports how often the actual price falls inside the range (should be ~80%), the
range's width, and both by price band, source and age, so the band can be
judged the way a buyer would use it. Needs the same data files, --scrape-year
and --random-state the models were trained with.

Usage:
    python Code/evaluate_range.py data/*.csv --low models/price_low.joblib --high models/price_high.joblib
"""
import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

import train


def row(label, mask, y, pred, low, high):
    if not mask.any():
        return '{:<24} {:>9,}'.format(label, 0)
    inside = (y[mask] >= low[mask]) & (y[mask] <= high[mask])
    width = (high[mask] - low[mask])
    return '{:<24} {:>9,} {:>8.1%} {:>8.1%} {:>8.1%} {:>10,.0f} {:>8.1%}'.format(
        label, int(mask.sum()), inside.mean(), (y[mask] < low[mask]).mean(), (y[mask] > high[mask]).mean(),
        np.median(width), np.median(width / pred[mask]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', nargs='+')
    parser.add_argument('--model', default='models/price_model.joblib')
    parser.add_argument('--low', default='models/price_low.joblib')
    parser.add_argument('--high', default='models/price_high.joblib')
    parser.add_argument('--scrape-year', type=int, default=2020)
    parser.add_argument('--vin-specs', default='data/vin_specs.csv')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=10)
    parser.add_argument('--cache', help='directory to keep each model\'s test predictions in (point.npy, low.npy, high.npy), '
                        'so a re-run after a crash or with a new model only scores what changed')
    args = parser.parse_args()

    data = train.clean(train.load_data(args.data, random_state=args.random_state), args.scrape_year, vin_specs=args.vin_specs)
    y = data.pop('Price')
    X = data[train.CATEGORICAL + train.NUMERIC]
    _, X_test, _, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    print('{:,} test cars'.format(len(X_test)), flush=True)

    def predictions(name, model_file):
        cached = os.path.join(args.cache, name + '.npy') if args.cache else None
        if cached and os.path.exists(cached):
            return np.load(cached)
        pred = joblib.load(model_file).predict(X_test)
        if cached:
            os.makedirs(args.cache, exist_ok=True)
            np.save(cached, pred)
        print('scored {}'.format(name), flush=True)
        return pred

    pred = predictions('point', args.model)
    low = np.minimum(predictions('low', args.low), pred)
    high = np.maximum(predictions('high', args.high), pred)
    y = y_test.to_numpy(dtype=float)

    header = '{:<24} {:>9} {:>8} {:>8} {:>8} {:>10} {:>8}'.format(
        'segment', 'cars', 'inside', 'below', 'above', 'width($)', 'width/est')
    print(header)
    print(row('all', np.ones(len(y), bool), y, pred, low, high))
    for lo, hi in [(500, 5000), (5000, 10000), (10000, 20000), (20000, 35000), (35000, 60000), (60000, 500000)]:
        print(row('price ${:,}-{:,}'.format(lo, hi), (y >= lo) & (y < hi), y, pred, low, high))
    src = X_test['Source'].astype(str).to_numpy()
    for s in ['craigslist', 'absee:carmax', 'absee:toyota', 'absee:santander', 'absee:carvana', 'classiccars']:
        print(row(s, src == s, y, pred, low, high))
    age = X_test['Age'].to_numpy()
    for lo, hi in [(0, 3), (3, 8), (8, 15), (15, 200)]:
        print(row('age {}-{} yrs'.format(lo, hi), (age >= lo) & (age < hi), y, pred, low, high))
    # how the width itself relates to accuracy: narrow ranges should be the reliable ones
    rel = (high - low) / pred
    print('\nby range width (width / estimate):')
    print(header)
    edges = [0, .2, .3, .4, .6, 1, np.inf]
    for lo, hi in zip(edges, edges[1:]):
        print(row('width {:.0%}-{:.0%}'.format(lo, hi) if hi != np.inf else 'width over {:.0%}'.format(lo),
                  (rel >= lo) & (rel < hi), y, pred, low, high))


if __name__ == '__main__':
    sys.exit(main())
