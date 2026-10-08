"""Sanity tests for the scraper, cleaning and evaluation helpers.

Run from the repo root with:  python -m pytest tests
"""
import glob
import html
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


# Hand-written pages shaped like the current cars.com results markup (vehicle cards
# carrying a data-vehicle-details JSON attribute). They are not captured from the
# live site; run `python Code/scraper.py --check <zip>` to confirm against it.

def vehicle_card(details, location='Brooklyn, NY (3 mi.)', price_text=None, mileage_text=None, title=None):
    attr = html.escape(json.dumps(details), quote=True)
    return ('<div class="vehicle-card" data-listing-id="{id}">'
            '<a class="vehicle-card-link" data-vehicle-details="{attr}"><h2 class="title">{title}</h2></a>'
            '<span class="primary-price">{price}</span><div class="mileage">{mileage}</div>'
            '<div class="miles-from">{location}</div></div>').format(
        id=details.get('vin', ''), attr=attr, title=title or '', price=price_text or '',
        mileage=mileage_text or '', location=location)


def results_page(*cards):
    return '<html><body><div class="vehicle-cards">' + ''.join(cards) + '</div></body></html>'


def details(vin, make='Toyota', model='Corolla', year='2018', price='24990', mileage='35000', **extra):
    d = {'vin': vin, 'make': make, 'model': model, 'year': year, 'price': price, 'mileage': mileage,
         'bodyStyle': 'Sedan', 'trim': 'LE', 'exteriorColor': 'Blue', 'stockType': 'used'}
    d.update(extra)
    return d


# --- scraper ---------------------------------------------------------------

def test_parse_listings_reads_vehicle_details_json():
    [car] = scraper.parse_listings(results_page(vehicle_card(details('VIN1'))))
    assert car == {
        'Price': 24990, 'Make': 'Toyota', 'Model': 'Corolla', 'Year': 2018, 'Body Style': 'Sedan',
        'City': 'Brooklyn', 'State': 'NY', 'Milage': 35000, 'Color': 'Blue', 'Trim': 'LE', 'VIN': 'VIN1',
    }


def test_parse_listings_fills_gaps_from_card_text():
    sparse = details('VIN2', price=None, mileage=None, year=None, exteriorColor=None)
    card = vehicle_card(sparse, location='St. Louis, MO (12 mi.)', price_text='$18,750',
                        mileage_text='61,204 mi.', title='Used 2016 Toyota Corolla S')
    [car] = scraper.parse_listings(results_page(card))
    assert (car['Price'], car['Milage'], car['Year']) == (18750, 61204, 2016)
    assert (car['City'], car['State']) == ('St. Louis', 'MO')
    assert car['Color'] is None


def test_parse_listings_drops_incomplete_and_duplicate_cards():
    page = results_page(
        vehicle_card(details('VIN3')),
        vehicle_card(details('VIN3')),  # same car rendered twice
        vehicle_card(details('VIN4', price=None)),  # "Not priced" listing
        '<div data-vehicle-details="{not json"></div>',
        vehicle_card(details('VIN5', make='Mazda', model='CX-5')),
    )
    assert [c['VIN'] for c in scraper.parse_listings(page)] == ['VIN3', 'VIN5']


def test_parse_listings_falls_back_to_json_ld():
    ld = {'@context': 'https://schema.org', '@graph': [{
        '@type': 'Car', 'brand': {'@type': 'Brand', 'name': 'Honda'}, 'model': 'Civic',
        'vehicleModelDate': '2017', 'bodyType': 'Sedan', 'color': 'Gray', 'vehicleIdentificationNumber': 'VIN6',
        'mileageFromOdometer': {'@type': 'QuantitativeValue', 'value': '42,000', 'unitCode': 'SMI'},
        'offers': {'@type': 'Offer', 'price': '16995', 'seller': {'address': {
            'addressLocality': 'Austin', 'addressRegion': 'TX'}}},
    }]}
    page = '<html><script type="application/ld+json">{}</script></html>'.format(json.dumps(ld))
    [car] = scraper.parse_listings(page)
    assert car == {
        'Price': 16995, 'Make': 'Honda', 'Model': 'Civic', 'Year': 2017, 'Body Style': 'Sedan',
        'City': 'Austin', 'State': 'TX', 'Milage': 42000, 'Color': 'Gray', 'Trim': None, 'VIN': 'VIN6',
    }


def test_parse_listings_on_unrecognized_page_returns_nothing():
    assert scraper.parse_listings('<html><body><p>No results</p></body></html>') == []


@pytest.mark.parametrize('value, expected', [
    ('$24,990', 24990), ('35,123 mi.', 35123), (2019, 2019), (18500.0, 18500),
    ('Not Priced', None), (None, None), ('', None),
])
def test_to_number(value, expected):
    assert scraper.to_number(value) == expected


def test_search_params_target_used_cars_sorted_by_distance():
    params = scraper.search_params(10001, 3)
    assert params['zip'] == 10001 and params['page'] == 3 and params['page_size'] == 100
    assert params['stock_type'] == 'used' and params['sort'] == 'distance'


def test_is_blocked():
    assert scraper.is_blocked('<html><title>Pardon Our Interruption</title></html>')
    assert scraper.is_blocked('<div id="px-captcha"></div>')
    assert not scraper.is_blocked(results_page(vehicle_card(details('VIN7'))))


def fake_fetch_from(pages, calls):
    def fake_fetch(session, zip_code, page_num, radius='all'):
        calls.append((zip_code, page_num))
        page = pages.get((zip_code, page_num), results_page())
        if isinstance(page, Exception):
            raise page
        return page
    return fake_fetch


def test_scrape_zips_dedupes_and_stops_on_empty_page(tmp_path, monkeypatch):
    pages = {
        (10001, 1): results_page(vehicle_card(details('A')), vehicle_card(details('B', make='Mazda'))),
        (10001, 2): results_page(vehicle_card(details('A'))),
        (10002, 1): results_page(vehicle_card(details('C', make='Ford', model='Focus'))),
    }
    calls = []
    monkeypatch.setattr(scraper, 'fetch_page', fake_fetch_from(pages, calls))
    out = tmp_path / 'carlist1.csv'
    assert scraper.scrape_zips([10001, 10002], str(out), max_pages=50, delay=0) == 3
    # each page fetched once, and pagination stops at the first empty page
    assert calls == [(10001, 1), (10001, 2), (10001, 3), (10002, 1), (10002, 2)]

    df = pd.read_csv(out)
    assert list(df.columns) == scraper.COLUMNS
    assert sorted(df['VIN']) == ['A', 'B', 'C']


def test_scrape_zips_stops_when_blocked_and_keeps_what_it_has(tmp_path, monkeypatch):
    pages = {
        (10001, 1): results_page(vehicle_card(details('A'))),
        (10001, 2): scraper.BlockedError('bot check'),
    }
    calls = []
    monkeypatch.setattr(scraper, 'fetch_page', fake_fetch_from(pages, calls))
    out = tmp_path / 'carlist1.csv'
    assert scraper.scrape_zips([10001, 10002], str(out), delay=0) == 1
    assert calls == [(10001, 1), (10001, 2)]  # no requests after being blocked
    assert list(pd.read_csv(out)['VIN']) == ['A']


def test_scraped_csv_survives_load_and_clean(tmp_path, monkeypatch):
    pages = {(10001, 1): results_page(vehicle_card(details('A')),
                                      vehicle_card(details('B', exteriorColor=None, bodyStyle=None)))}
    monkeypatch.setattr(scraper, 'fetch_page', fake_fetch_from(pages, []))
    scraper.scrape_zips([10001], str(tmp_path / 'carlist1.csv'), delay=0)
    cars = car_utils.clean(car_utils.load_raw(str(tmp_path / 'carlist*.csv')), reference_year=2020)
    assert sorted(cars['VIN']) == ['A', 'B']
    assert cars.set_index('VIN').loc['B', ['Color', 'Body Style']].tolist() == ['Unknown', 'Unknown']


def test_parse_file_cli(tmp_path, capsys, monkeypatch):
    page = tmp_path / 'page.html'
    page.write_text(results_page(vehicle_card(details('VIN8'))))
    monkeypatch.setattr(sys, 'argv', ['scraper.py', '--parse-file', str(page)])
    scraper.main()
    out = capsys.readouterr().out
    assert '1 listings parsed' in out and 'VIN8' in out


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
        (7000, 'Kia', 'Rio', 2012, None, 'Reno', 'NV', 90000, None, 'V6'),  # missing color/body: kept
        (6000, None, 'Rio', 2012, 'Sedan', 'Reno', 'NV', 90000, 'Red', 'V8'),  # missing make: dropped
        (15000, 'Mazda', '3', 2016, 'Sedan', 'Ogden', 'UT', 40000, 'Red', 'V7'),
    ]
    df = pd.DataFrame(rows, columns=car_utils.LEGACY_COLUMNS + ['VIN'])
    cleaned = car_utils.clean(df, reference_year=2020)
    assert sorted(cleaned['VIN']) == ['V1', 'V6', 'V7']
    assert cleaned.set_index('VIN').loc['V6', ['Color', 'Body Style']].tolist() == ['Unknown', 'Unknown']
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
