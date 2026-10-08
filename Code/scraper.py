"""Scrape used-car listings from cars.com search results.

Replaces the 20 near-identical ``split*.py`` scripts and ``multi_proceses2.py``:
the zip-code range is split into chunks and each chunk is scraped by its own
worker process, writing ``carlist<N>.csv`` into the output directory.

Example (same coverage as the old split1..split20 scripts):

    python scraper.py --zip-codes zip_code_database.csv --zip-start 10000 \
        --zip-end 100000 --workers 20 --out-dir data/raw
"""
import argparse
import csv
import logging
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup as bsoup

SEARCH_URL = ('https://www.cars.com/for-sale/searchresults.action/?page={page}&perPage=100'
              '&rd=99999&searchSource=GN_BREADCRUMB&sort=distance-nearest&zc={zip}')
HEADERS = {'User-Agent': 'Mozilla/5.0 (used-car-market research scraper)'}

COLUMNS = ['Price', 'Make', 'Model', 'Year', 'Body Style', 'City', 'State', 'Milage', 'Color', 'Trim', 'VIN']

# One regex per field, applied to a single listing block of the page's JSON.
FIELD_PATTERNS = {
    'Price': r',"price":(\d+),"mileage":',
    'Make': r'"make":"(.+?)","makeId"',
    'Model': r'"model":"(.+?)","modelId"',
    'Year': r'"year":(\d{4}),"trim"',
    'Body Style': r'"bodyStyle":"(.+?)","customerId"',
    'City': r'"city":"(.+?)","state":',
    'State': r',"state":"(.+?)","truncatedDescription',
    'Milage': r',"mileage":(\d+),"vin":',
    'Trim': r'"trim":"(.*?)"',
    'VIN': r'"vin":"(.+?)"',
}
REQUIRED = ['Price', 'Make', 'Model', 'Year', 'Body Style', 'City', 'State', 'Milage']
BLOCK_COLOR = r'","color":"(.+?)"},{"@context":"'
PAGE_COLOR = r'/","color":"(.*?)"},{"@context":"http://schema.org"'

log = logging.getLogger('scraper')


def fetch_page(session, zip_code, page_num, timeout=30):
    """Return the text of all <script> tags on one search-results page."""
    response = session.get(SEARCH_URL.format(page=page_num, zip=zip_code), headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    soup = bsoup(response.content, features='lxml')
    return str(soup.find_all('script'))


def parse_listings(page_text):
    """Turn the script text of one results page into a list of listing dicts.

    Listings missing any required field are dropped rather than written
    with shifted columns.
    """
    blocks = [b for b in re.split('"type"', page_text) if b.startswith(':"inventory"')]
    page_colors = re.findall(PAGE_COLOR, page_text)
    # The page-level color list can only be matched to listings by position,
    # so only trust it when it lines up one-to-one with the listing blocks.
    use_page_colors = len(page_colors) == len(blocks)

    listings = []
    for i, block in enumerate(blocks):
        car = {}
        for field, pattern in FIELD_PATTERNS.items():
            match = re.search(pattern, block)
            car[field] = match.group(1) if match else None
        if any(car[field] is None for field in REQUIRED):
            continue
        block_color = re.search(BLOCK_COLOR, block)
        if block_color:
            car['Color'] = block_color.group(1)
        else:
            car['Color'] = page_colors[i] if use_page_colors else None
        listings.append(car)
    return listings


def scrape_zips(zip_codes, out_path, max_pages=50, delay=1.0):
    """Scrape every page for each zip code and append unique listings to ``out_path``."""
    seen = set()
    write_header = not os.path.exists(out_path)
    with requests.Session() as session, open(out_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for zip_code in zip_codes:
            for page_num in range(1, max_pages + 1):
                try:
                    listings = parse_listings(fetch_page(session, zip_code, page_num))
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


def _run_chunk(args):
    zip_codes, out_path, max_pages, delay = args
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)s %(message)s')
    return scrape_zips(zip_codes, out_path, max_pages, delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zip-codes', default='zip_code_database.csv', help="CSV with a 'zip' column")
    parser.add_argument('--zip-start', type=int, default=10000)
    parser.add_argument('--zip-end', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--max-pages', type=int, default=50, help='cars.com shows at most 50 pages per search')
    parser.add_argument('--delay', type=float, default=1.0, help='seconds to wait between requests per worker')
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()

    zips = pd.read_csv(args.zip_codes)['zip'].to_numpy()
    zips = zips[(zips >= args.zip_start) & (zips < args.zip_end)]
    os.makedirs(args.out_dir, exist_ok=True)
    chunks = [
        (chunk, os.path.join(args.out_dir, 'carlist{}.csv'.format(i + 1)), args.max_pages, args.delay)
        for i, chunk in enumerate(np.array_split(zips, args.workers))
    ]
    with Pool(processes=args.workers) as pool:
        totals = pool.map(_run_chunk, chunks)
    print('Scraped {} unique cars into {}'.format(sum(totals), args.out_dir))


if __name__ == '__main__':
    main()
