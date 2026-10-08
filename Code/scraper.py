"""Scrape used-car listings from cars.com search results.

The zip-code range is split into chunks and each chunk is scraped by its own
worker process, writing ``carlist<N>.csv`` into the output directory.

Example (same coverage as the original 20 split scripts):

    python scraper.py --zip-codes zip_code_database.csv --zip-start 10000 \
        --zip-end 100000 --workers 20 --out-dir data/raw

Check that parsing still works against the live site, or against a page saved
from your browser, before starting a long run:

    python scraper.py --check 10001 --save-html page.html
    python scraper.py --parse-file page.html

Listings are read, in order of preference, from:
  1. the JSON in each result card's ``data-vehicle-details`` attribute,
  2. schema.org JSON-LD ``<script type="application/ld+json">`` blocks,
with the card's visible text filling any fields the JSON is missing.
"""
import argparse
import csv
import json
import logging
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup as bsoup

SEARCH_URL = 'https://www.cars.com/shopping/results/'
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'),
    'Accept-Language': 'en-US,en;q=0.9',
}

COLUMNS = ['Price', 'Make', 'Model', 'Year', 'Body Style', 'City', 'State', 'Milage', 'Color', 'Trim', 'VIN']
# A listing without these is useless for the price model; everything else may be blank.
REQUIRED = ['Price', 'Make', 'Model', 'Year', 'Milage']

# Key names cars.com has used for each field in its listing JSON, in order of preference.
DETAIL_KEYS = {
    'Price': ['price', 'listPrice', 'currentPrice'],
    'Make': ['make', 'makeName'],
    'Model': ['model', 'modelName'],
    'Year': ['year', 'modelYear'],
    'Body Style': ['bodyStyle', 'bodyType'],
    'City': ['city', 'dealerCity'],
    'State': ['state', 'dealerState'],
    'Milage': ['mileage', 'miles', 'odometer'],
    'Color': ['exteriorColor', 'color', 'exterior_color'],
    'Trim': ['trim', 'trimName'],
    'VIN': ['vin'],
}
NUMERIC = {'Price', 'Year', 'Milage'}

# Card text used when the JSON lacks a field.
CARD_SELECTORS = {
    'Price': ['.primary-price', '[data-qa="primary-price"]'],
    'Milage': ['.mileage', '[data-qa="mileage"]'],
    'Location': ['.miles-from', '[data-qa="miles-from"]', '.dealer-location'],
    'Title': ['.title', 'h2'],
}
LOCATION_RE = re.compile(r'([A-Za-z .\'-]+),\s*([A-Z]{2})\b')
BLOCK_MARKERS = ('captcha', 'pardon our interruption', 'access denied', 'are you a robot', 'px-captcha')

log = logging.getLogger('scraper')


class BlockedError(RuntimeError):
    """cars.com answered with a bot-check or access-denied page instead of results."""


def search_params(zip_code, page_num, page_size=100, radius='all'):
    return {
        'stock_type': 'used',
        'zip': zip_code,
        'maximum_distance': radius,
        'sort': 'distance',
        'page': page_num,
        'page_size': page_size,
    }


def fetch_page(session, zip_code, page_num, radius='all', timeout=30, retries=3):
    """Return the HTML of one search-results page.

    Retries 429 and 5xx responses with exponential backoff, and raises
    BlockedError when the site serves a bot check instead of results.
    """
    for attempt in range(retries + 1):
        response = session.get(SEARCH_URL, params=search_params(zip_code, page_num, radius=radius),
                               headers=HEADERS, timeout=timeout)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
        if response.status_code == 403:
            raise BlockedError('403 Forbidden for {}'.format(response.url))
        response.raise_for_status()
        html = response.text
        if is_blocked(html):
            raise BlockedError('bot check served for {}'.format(response.url))
        return html


def is_blocked(html):
    head = html[:20000].lower()
    return 'vehicle-details' not in head and any(marker in head for marker in BLOCK_MARKERS)


def to_number(value):
    """'$24,990' -> 24990, '35,123 mi.' -> 35123, 2019 -> 2019; None when there is no number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    digits = re.sub(r'[^\d.]', '', str(value)).split('.')[0]
    return int(digits) if digits else None


def _first(mapping, keys):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, '', [], {}):
            return value
    return None


def _clean(field, value):
    if value is None:
        return None
    if isinstance(value, dict):  # e.g. {"name": "Toyota"}
        value = _first(value, ['name', 'value'])
    if field in NUMERIC:
        return to_number(value)
    value = str(value).strip()
    return value or None


def from_details(details):
    """Map one ``data-vehicle-details`` JSON object to our columns."""
    return {field: _clean(field, _first(details, keys)) for field, keys in DETAIL_KEYS.items()}


def from_json_ld(item):
    """Map one schema.org Car/Vehicle object to our columns."""
    offers = item.get('offers') or {}
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    seller = offers.get('seller') or {}
    address = seller.get('address') or {}
    mileage = item.get('mileageFromOdometer')
    return {
        'Price': _clean('Price', offers.get('price')),
        'Make': _clean('Make', _first(item, ['brand', 'manufacturer'])),
        'Model': _clean('Model', item.get('model')),
        'Year': _clean('Year', _first(item, ['vehicleModelDate', 'modelDate', 'productionDate'])),
        'Body Style': _clean('Body Style', item.get('bodyType')),
        'City': _clean('City', address.get('addressLocality')),
        'State': _clean('State', address.get('addressRegion')),
        'Milage': _clean('Milage', mileage.get('value') if isinstance(mileage, dict) else mileage),
        'Color': _clean('Color', item.get('color')),
        'Trim': _clean('Trim', item.get('vehicleConfiguration')),
        'VIN': _clean('VIN', item.get('vehicleIdentificationNumber')),
    }


def _card_text(card, field):
    for selector in CARD_SELECTORS[field]:
        node = card.select_one(selector)
        if node and node.get_text(strip=True):
            return node.get_text(' ', strip=True)
    return None


def _card_root(node):
    """The whole result card around the element carrying data-vehicle-details."""
    for candidate in [node] + list(node.parents):
        if getattr(candidate, 'attrs', None) is None:
            break
        classes = candidate.get('class') or []
        if candidate.has_attr('data-listing-id') or any('vehicle-card' == c for c in classes):
            return candidate
    return node


def _fill_from_card(car, card):
    if car['Price'] is None:
        car['Price'] = to_number(_card_text(card, 'Price'))
    if car['Milage'] is None:
        car['Milage'] = to_number(_card_text(card, 'Milage'))
    if car['City'] is None or car['State'] is None:
        match = LOCATION_RE.search(_card_text(card, 'Location') or '')
        if match:
            car['City'] = car['City'] or match.group(1).strip()
            car['State'] = car['State'] or match.group(2)
    if car['Year'] is None:
        match = re.match(r'\s*(?:Used\s+|Certified\s+)?((?:19|20)\d{2})\b', _card_text(card, 'Title') or '')
        if match:
            car['Year'] = int(match.group(1))


def _iter_json_ld(soup):
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string or '')
        except ValueError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            node = stack.pop(0)
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                types = node.get('@type')
                types = types if isinstance(types, list) else [types]
                if any(t in ('Car', 'Vehicle', 'Product') for t in types) and node.get('offers'):
                    yield node
                stack.extend(v for k, v in node.items() if k in ('@graph', 'itemListElement', 'item'))


def parse_listings(html):
    """Turn one search-results page into a list of listing dicts with COLUMNS keys.

    Listings missing a REQUIRED field are dropped.
    """
    soup = bsoup(html, features='lxml')
    cars = []
    for card in soup.select('[data-vehicle-details]'):
        try:
            details = json.loads(card['data-vehicle-details'])
        except ValueError:
            log.debug('unparseable data-vehicle-details: %.200s', card['data-vehicle-details'])
            continue
        car = from_details(details)
        _fill_from_card(car, _card_root(card))
        cars.append(car)
    if not cars:
        cars = [from_json_ld(item) for item in _iter_json_ld(soup)]

    listings, seen = [], set()
    for car in cars:
        if any(car[field] is None for field in REQUIRED):
            continue
        key = car['VIN'] or tuple(car[c] for c in COLUMNS)
        if key not in seen:  # the same listing can appear in several page elements
            seen.add(key)
            listings.append(car)
    return listings


def scrape_zips(zip_codes, out_path, max_pages=50, delay=1.0, radius='all'):
    """Scrape every page for each zip code and append unique listings to ``out_path``.

    Stops early, keeping what was already written, if the site starts serving bot checks.
    """
    seen = set()
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with requests.Session() as session, open(out_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for zip_code in zip_codes:
            for page_num in range(1, max_pages + 1):
                try:
                    listings = parse_listings(fetch_page(session, zip_code, page_num, radius=radius))
                except BlockedError as e:
                    log.error('%s: stopping, blocked by cars.com (%s)', out_path, e)
                    return len(seen)
                except requests.RequestException as e:
                    log.warning('zip %s page %s failed: %s', zip_code, page_num, e)
                    time.sleep(delay * 5)
                    continue
                finally:
                    time.sleep(delay)
                if not listings:
                    break
                for car in listings:
                    key = car['VIN'] or tuple(car[c] for c in COLUMNS)
                    if key not in seen:
                        seen.add(key)
                        writer.writerow(car)
            f.flush()
            log.info('%s: zip %s done, %d unique cars', out_path, zip_code, len(seen))
    return len(seen)


def report(listings, source):
    """Print how many listings parsed and how often each field was found."""
    print('{}: {} listings parsed'.format(source, len(listings)))
    if not listings:
        print('  Nothing parsed - the page layout may have changed. Save the page with --save-html '
              'and check for data-vehicle-details attributes or application/ld+json scripts.')
        return
    for field in COLUMNS:
        found = sum(car[field] is not None for car in listings)
        print('  {:<11} {:>4}/{} found'.format(field, found, len(listings)))
    for car in listings[:3]:
        print('  ', car)


def _run_chunk(args):
    zip_codes, out_path, max_pages, delay, radius = args
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)s %(message)s')
    return scrape_zips(zip_codes, out_path, max_pages, delay, radius)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zip-codes', default='zip_code_database.csv', help="CSV with a 'zip' column")
    parser.add_argument('--zip-start', type=int, default=10000)
    parser.add_argument('--zip-end', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--max-pages', type=int, default=50)
    parser.add_argument('--radius', default='all', help="search radius in miles, or 'all'")
    parser.add_argument('--delay', type=float, default=1.0, help='seconds to wait between requests per worker')
    parser.add_argument('--out-dir', default='.')
    parser.add_argument('--check', metavar='ZIP', help='fetch one results page for ZIP, report what parsed, and exit')
    parser.add_argument('--save-html', metavar='PATH', help='with --check, also save the fetched page')
    parser.add_argument('--parse-file', metavar='PATH', help='parse a saved results page, report, and exit')
    args = parser.parse_args()

    if args.parse_file:
        with open(args.parse_file, encoding='utf-8', errors='replace') as f:
            report(parse_listings(f.read()), args.parse_file)
        return
    if args.check:
        with requests.Session() as session:
            html = fetch_page(session, args.check, 1, radius=args.radius)
        if args.save_html:
            with open(args.save_html, 'w', encoding='utf-8') as f:
                f.write(html)
        report(parse_listings(html), 'zip {} page 1'.format(args.check))
        return

    zips = pd.read_csv(args.zip_codes)['zip'].to_numpy()
    zips = zips[(zips >= args.zip_start) & (zips < args.zip_end)]
    os.makedirs(args.out_dir, exist_ok=True)
    chunks = [
        (chunk, os.path.join(args.out_dir, 'carlist{}.csv'.format(i + 1)), args.max_pages, args.delay, args.radius)
        for i, chunk in enumerate(np.array_split(zips, args.workers))
    ]
    with Pool(processes=args.workers) as pool:
        totals = pool.map(_run_chunk, chunks)
    print('Scraped {} unique cars into {}'.format(sum(totals), args.out_dir))


if __name__ == '__main__':
    main()
