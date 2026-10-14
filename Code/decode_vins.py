"""Decode VINs with NHTSA's free vPIC service into trim and engine specs.

Reads the VIN column of the given CSVs, decodes every valid 17-character VIN not
already in the output (so it can be stopped and resumed), 50 per request, and
writes data/vin_specs.csv, which train.py joins onto the listings.

Usage:
    python Code/decode_vins.py data/craigslist_vehicles.csv data/classiccars_vehicles.csv data/gsa_vehicles.csv
"""
import argparse
import csv
import os
import re
import time

import pandas as pd

from collect_common import session

FIELDS = ['VIN', 'Make', 'Model', 'ModelYear', 'Trim', 'Series', 'BodyClass', 'DriveType', 'EngineCylinders',
          'DisplacementL', 'EngineHP', 'FuelTypePrimary', 'TransmissionStyle', 'Doors', 'GVWR', 'ErrorCode']
VALID = re.compile(r'[A-HJ-NPR-Z0-9]{17}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', nargs='+', help='CSV files with a VIN column')
    parser.add_argument('--out', default='data/vin_specs.csv')
    parser.add_argument('--delay', type=float, default=1.0, help='seconds between requests')
    args = parser.parse_args()

    vins = set()
    for path in args.data:
        col = pd.read_csv(path, usecols=['VIN'], dtype=str)['VIN'].dropna().str.strip().str.upper()
        vins.update(col[col.str.fullmatch(VALID)])
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out, usecols=['VIN'], dtype=str)['VIN'])
    todo = sorted(vins - done)
    print('{:,} unique VINs, {:,} already decoded, {:,} to do'.format(len(vins), len(done), len(todo)), flush=True)

    http = session()
    with open(args.out, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if not done:
            writer.writeheader()
        for i in range(0, len(todo), 50):
            batch = todo[i:i + 50]
            for attempt in range(4):
                try:
                    r = http.post('https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/',
                                  data={'format': 'json', 'data': ';'.join(batch)}, timeout=120)
                    r.raise_for_status()
                    results = r.json()['Results']
                    break
                except Exception as e:   # outages happen; wait and retry
                    wait = 30 * (attempt + 1)
                    print('request failed ({}), retrying in {}s'.format(type(e).__name__, wait), flush=True)
                    time.sleep(wait)
            else:
                print('giving up for now; rerun to resume')
                return
            for res in results:
                writer.writerow({k: (res.get(k) or '') for k in FIELDS})
            f.flush()
            if (i // 50) % 100 == 0:
                print('{:,} / {:,}'.format(i + len(batch), len(todo)), flush=True)
            time.sleep(args.delay)
    print('done: {}'.format(args.out))


if __name__ == '__main__':
    main()
