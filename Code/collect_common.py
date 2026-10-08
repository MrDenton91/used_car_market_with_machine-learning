"""Helpers shared by the collect_*.py data collectors."""
import csv
import os
import time

import requests

# identify the collector honestly in every request
USER_AGENT = ('used-car-ml-research/1.0 '
              '(+https://github.com/MrDenton91/used_car_market_with_machine-learning)')

# the columns train.py reads come first; the rest are kept for analysis
COLUMNS = ['Price', 'Make', 'Model', 'Year', 'Body Style', 'City', 'State', 'Milage', 'Color',
           'Date Scraped', 'Source', 'Listing ID', 'VIN', 'Trim', 'Transmission', 'Engine',
           'Drive Type', 'Fuel Type', 'Interior Color', 'Title Status', 'Status', 'Auction End', 'URL']


def session():
    s = requests.Session()
    s.headers['User-Agent'] = USER_AGENT
    return s


def write_rows(path, rows):
    """Write rows (dicts) to a CSV with a header, replacing the file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def decode_vins(vins, http=None, delay=1.0):
    """Look up 17-character VINs with NHTSA's free vPIC API, 50 per request.

    Returns {vin: {'Body Style', 'Drive Type', 'Fuel Type', 'Engine'}}; VINs it
    can't decode (e.g. pre-1981 cars) are left out.
    """
    http = http or session()
    vins = sorted({v.strip().upper() for v in vins if v and len(v.strip()) == 17})
    decoded = {}
    for i in range(0, len(vins), 50):
        batch = vins[i:i + 50]
        r = http.post('https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/',
                      data={'format': 'json', 'data': ';'.join(batch)}, timeout=60)
        r.raise_for_status()
        for res in r.json()['Results']:
            if not res.get('Make'):
                continue
            cylinders = res.get('EngineCylinders')
            displacement = res.get('DisplacementL')
            engine = ' '.join(filter(None, [
                '{:.1f}L'.format(float(displacement)) if displacement else '',
                '{} cyl'.format(cylinders) if cylinders else '']))
            decoded[res['VIN']] = {
                'Body Style': res.get('BodyClass') or '',
                'Drive Type': res.get('DriveType') or '',
                'Fuel Type': res.get('FuelTypePrimary') or '',
                'Engine': engine,
            }
        time.sleep(delay)
    return decoded


def enrich_with_vins(rows, http=None):
    """Fill empty spec columns from the VIN decoder; never overwrite listed values."""
    decoded = decode_vins([r.get('VIN', '') for r in rows], http)
    for row in rows:
        specs = decoded.get((row.get('VIN') or '').strip().upper())
        if not specs:
            continue
        for key, value in specs.items():
            if value and not row.get(key):
                row[key] = value
    return len(decoded)
