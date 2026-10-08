"""Sanity tests for the scraper, cleaning and evaluation helpers.

Run from the repo root with:  python -m pytest tests
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'Code'))

import car_utils  # noqa: E402
import scraper  # noqa: E402


def listing_block(price, make, model, year, mileage, vin, color=None, city='Brooklyn', state='NY'):
    """One inventory block shaped like the cars.com page JSON the regexes expect."""
    block = ('"type":"inventory","make":"{make}","makeId":1,"model":"{model}","modelId":2,'
             '"year":{year},"trim":"Base","bodyStyle":"Sedan","customerId":3,'
             '"city":"{city}","state":"{state}","truncatedDescription":"x",'
             '"price":{price},"mileage":{mileage},"vin":"{vin}"').format(**locals())
    if color:
        block += '","color":"{}"}},{{"@context":"'.format(color)
    return block


def fake_page(*blocks):
    return '[<script>' + ','.join(blocks) + '</script>]'


# --- scraper ---------------------------------------------------------------

def test_parse_listings_extracts_fields():
    page = fake_page(listing_block(24990, 'Toyota', 'Corolla', 2018, 35000, 'VIN1', color='Blue'))
    [car] = scraper.parse_listings(page)
    assert car == {
        'Price': '24990', 'Make': 'Toyota', 'Model': 'Corolla', 'Year': '2018', 'Body Style': 'Sedan',
        'City': 'Brooklyn', 'State': 'NY', 'Milage': '35000', 'Color': 'Blue', 'Trim': 'Base', 'VIN': 'VIN1',
    }


def test_parse_listings_drops_incomplete_and_never_misaligns_colors():
    incomplete = listing_block(10000, 'Ford', 'Fiesta', 2015, 80000, 'VIN2').replace('"price":10000,', '')
    page = fake_page(
        listing_block(15000, 'Honda', 'Civic', 2016, 60000, 'VIN3'),
        incomplete,
        listing_block(20000, 'Mazda', 'CX-5', 2019, 20000, 'VIN4'),
    )
    cars = scraper.parse_listings(page)
    assert [c['VIN'] for c in cars] == ['VIN3', 'VIN4']
    # no per-listing color and no one-to-one page color list -> unknown, not someone else's color
    assert all(c['Color'] is None for c in cars)


def test_scrape_zips_dedupes_and_stops_on_empty_page(tmp_path, monkeypatch):
    pages = {
        (10001, 1): fake_page(listing_block(15000, 'Honda', 'Civic', 2016, 60000, 'A', color='Red'),
                              listing_block(20000, 'Mazda', 'CX-5', 2019, 20000, 'B', color='Gray')),
        (10001, 2): fake_page(listing_block(15000, 'Honda', 'Civic', 2016, 60000, 'A', color='Red')),
        (10002, 1): fake_page(listing_block(9000, 'Ford', 'Focus', 2012, 90000, 'C', color='White')),
    }
    calls = []

    def fake_fetch(session, zip_code, page_num):
        calls.append((zip_code, page_num))
        return pages.get((zip_code, page_num), '[]')

    monkeypatch.setattr(scraper, 'fetch_page', fake_fetch)
    out = tmp_path / 'carlist1.csv'
    assert scraper.scrape_zips([10001, 10002], str(out), max_pages=50, delay=0) == 3
    # each page fetched once, and pagination stops at the first empty page
    assert calls == [(10001, 1), (10001, 2), (10001, 3), (10002, 1), (10002, 2)]

    df = pd.read_csv(out)
    assert list(df.columns) == scraper.COLUMNS
    assert sorted(df['VIN']) == ['A', 'B', 'C']


# --- loading and cleaning --------------------------------------------------

def test_load_raw_reads_legacy_and_new_files(tmp_path):
    (tmp_path / 'carlist1.csv').write_text('24990,Toyota,Corolla,2018,Sedan,Brooklyn,NY,35000,Blue\n')
    (tmp_path / 'carlist2.csv').write_text(
        ','.join(scraper.COLUMNS) + '\n15000,Honda,Civic,2016,Sedan,Austin,TX,60000,Red,LX,VIN9\n')
    df = car_utils.load_raw(str(tmp_path / 'carlist*.csv'))
    assert len(df) == 2
    assert set(car_utils.LEGACY_COLUMNS) <= set(df.columns)
    assert df.loc[df['Make'] == 'Honda', 'VIN'].item() == 'VIN9'


def test_load_raw_errors_when_nothing_matches(tmp_path):
    with pytest.raises(FileNotFoundError):
        car_utils.load_raw(str(tmp_path / 'nope*.csv'))


def test_clean_filters_bad_rows_and_dedupes_by_vin():
    rows = [
        # Price, Make, Model, Year, Body, City, State, Milage, Color, VIN
        (20000, 'Toyota', 'Camry', 2018, 'Sedan', 'Austin', 'TX', 30000, 'Blue', 'V1'),
        (19500, 'Toyota', 'Camry', 2018, 'Sedan', 'Dallas', 'TX', 30100, 'Blue', 'V1'),  # relisted car
        (100_000_000, 'Ford', 'F-150', 2017, 'Pickup', 'Waco', 'TX', 50000, 'Red', 'V2'),  # bogus price
        (1, 'Ford', 'F-150', 2017, 'Pickup', 'Waco', 'TX', 50000, 'Red', 'V3'),  # bogus price
        (8000, 'Honda', 'Civic', 2010, 'Sedan', 'Reno', 'NV', 5_000_000, 'Gray', 'V4'),  # bogus mileage
        (9000, 'Honda', 'Civic', 2011, 'Sedan', 'Reno', 'NV', 'n/a', 'Gray', 'V5'),  # unparseable
        (7000, 'Kia', 'Rio', 2012, 'Sedan', 'Reno', 'NV', 90000, None, 'V6'),  # missing color
        (15000, 'Mazda', '3', 2016, 'Sedan', 'Ogden', 'UT', 40000, 'Red', 'V7'),
    ]
    df = pd.DataFrame(rows, columns=car_utils.LEGACY_COLUMNS + ['VIN'])
    cleaned = car_utils.clean(df, reference_year=2020)
    assert sorted(cleaned['VIN']) == ['V1', 'V7']
    assert cleaned.set_index('VIN').loc['V7', 'Age'] == 4


def test_clean_without_vin_drops_exact_duplicates():
    row = (20000, 'Toyota', 'Camry', 2018, 'Sedan', 'Austin', 'TX', 30000, 'Blue')
    df = pd.DataFrame([row, row], columns=car_utils.LEGACY_COLUMNS)
    assert len(car_utils.clean(df, reference_year=2020)) == 1


# --- evaluation ------------------------------------------------------------

class FixedModel:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, X):
        return self.predictions


def test_evaluate_reports_rmse_not_mse():
    y = np.array([10.0, 20.0, 30.0])
    scores = car_utils.evaluate(FixedModel([13.0, 24.0, 30.0]), None, y, verbose=False)
    assert scores['rmse'] == pytest.approx(np.sqrt((9 + 16 + 0) / 3))
    assert scores['mae'] == pytest.approx(7 / 3)
    assert scores['baseline_rmse'] == pytest.approx(np.sqrt((100 + 0 + 100) / 3))
    assert scores['normalized_rmse'] == pytest.approx(scores['rmse'] / scores['baseline_rmse'])


def test_evaluate_accepts_keras_style_column_predictions(capsys):
    y = pd.Series([10.0, 20.0, 30.0])
    scores = car_utils.evaluate(FixedModel([[10.0], [20.0], [30.0]]), None, y)
    assert scores['rmse'] == 0
    assert 'RMSE of model: 0.00' in capsys.readouterr().out


# --- end to end ------------------------------------------------------------

def synthetic_cars(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    makes = {'Toyota': 1.0, 'Honda': 0.95, 'BMW': 1.6, 'Ford': 0.9}
    make = rng.choice(list(makes), n)
    year = rng.integers(2005, 2021, n)
    mileage = rng.integers(1000, 200_000, n)
    price = (40000 * np.array([makes[m] for m in make]) * 0.9 ** (2020 - year)
             - 0.05 * mileage + rng.normal(0, 1500, n)).clip(1000)
    return pd.DataFrame({
        'Price': price.round(), 'Make': make, 'Model': 'Base', 'Year': year, 'Body Style': 'Sedan',
        'City': rng.choice(['Austin', 'Reno', 'Ogden'], n), 'State': 'TX', 'Milage': mileage,
        'Color': rng.choice(['Blue', 'Red'], n), 'VIN': ['VIN{}'.format(i) for i in range(n)],
    })


def test_pipeline_beats_baseline_on_synthetic_data():
    cars = car_utils.clean(synthetic_cars(), reference_year=2020)
    y = cars.pop('Price')
    X = pd.get_dummies(cars.drop(columns=['VIN', 'Year']), dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    baseline = car_utils.evaluate(DummyRegressor(strategy='median').fit(X_train, y_train),
                                  X_test, y_test, verbose=False)
    model = HistGradientBoostingRegressor(random_state=10).fit(X_train, y_train)
    scores = car_utils.evaluate(model, X_test, y_test, verbose=False)

    assert baseline['normalized_rmse'] == pytest.approx(1.0, abs=0.05)
    assert scores['normalized_rmse'] < 0.3
    assert scores['r2'] > 0.9


# --- repo hygiene ----------------------------------------------------------

NOTEBOOKS = glob.glob(os.path.join(REPO, '**', '*.ipynb'), recursive=True)
SECRET_PATTERNS = [r'\bAC[0-9a-f]{32}\b', r'authToken\s*=', r'twilio']


@pytest.mark.parametrize('path', NOTEBOOKS, ids=os.path.basename)
def test_notebooks_are_valid_and_free_of_secrets(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    json.loads(text)
    for pattern in SECRET_PATTERNS:
        assert not re.search(pattern, text, re.IGNORECASE), pattern


def test_python_sources_compile():
    for path in glob.glob(os.path.join(REPO, 'Code', '*.py')):
        with open(path) as f:
            compile(f.read(), path, 'exec')
