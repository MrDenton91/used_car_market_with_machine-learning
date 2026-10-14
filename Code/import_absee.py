"""Convert SEC ABS-EE auto-loan filings into vehicle records with real transaction values.

Lenders that securitize car loans (CarMax, Toyota, Honda, Ally, GM Financial,
Santander, ...) must file loan-level data with the SEC on Form ABS-EE. Every loan
lists the car's make, model, model year, new/used, type, the vehicle's value at
origination, and the buyer's state. The filings are public records.

Source: the parquet conversions published on Hugging Face by DenyTranDFW (one
dataset per trust, one folder per filing). download_absee() fetches the first
three filings of each trust, which hold the full initial loan pool; spot checks
against the original EDGAR XML matched exactly.

Usage:
    python Code/import_absee.py --download            # fetch filings into data/raw/absee (~1.5 GB)
    python Code/import_absee.py                       # convert to data/absee_vehicles.csv
"""
import argparse
import collections
import json
import os
import re
import time

import pandas as pd
import pyarrow.parquet as pq
import requests

from collect_common import COLUMNS, USER_AGENT

RAW = 'data/raw/absee'
VEHICLE_TYPES = {1: 'Car', 2: 'Truck', 3: 'SUV', 4: 'Motorcycle', 5: 'Van'}   # ABS-EE vehicleTypeCode
NEW_USED = {1: 'New', 2: 'Used'}
# some lenders abbreviate or qualify make names
MAKES = {'Chev': 'Chevrolet', 'Chevrole': 'Chevrolet', 'Chevrol': 'Chevrolet', 'Chevro': 'Chevrolet',
         'Chevrolet Truck': 'Chevrolet', 'Chevrolet Commercial': 'Chevrolet', 'Linc': 'Lincoln',
         'Dodg': 'Dodge', 'Dodge Truck': 'Dodge', 'Dodge Truc': 'Dodge', 'Dodge Commercial': 'Dodge',
         'Niss': 'Nissan', 'Nissan Medium Duty': 'Nissan', 'Hyun': 'Hyundai', 'Toyt': 'Toyota', 'Toyo': 'Toyota',
         'Buic': 'Buick', 'Hond': 'Honda', 'Mits': 'Mitsubishi', 'Mitsubis': 'Mitsubishi', 'Chry': 'Chrysler',
         'Cadi': 'Cadillac', 'Volk': 'Volkswagen', 'Volkswag': 'Volkswagen', 'Mazd': 'Mazda', 'Suba': 'Subaru',
         'Merz': 'Mercedes-Benz', 'Mercedes': 'Mercedes-Benz', 'Mercedes Benz': 'Mercedes-Benz', 'Merc': 'Mercury',
         'Acur': 'Acura', 'Lexs': 'Lexus', 'Infi': 'Infiniti', 'Scio': 'Scion', 'Pont': 'Pontiac', 'Mnni': 'Mini',
         'Volv': 'Volvo', 'Strn': 'Saturn', 'Suzi': 'Suzuki', 'Smrt': 'Smart', 'Lndr': 'Land Rover',
         'Land Rov': 'Land Rover', 'Humm': 'Hummer', 'Jagu': 'Jaguar', 'Gmc Light Duty': 'Gmc',
         'Ford Truck': 'Ford', 'Ford Commercial': 'Ford', 'Bmw': 'BMW', 'Gmc': 'GMC'}
STOP_WORDS = {'auto', 'automobile', 'financial', 'credit', 'lending', 'select', 'prime', 'llc'}
NEEDED = ['assetNumber', 'reportingPeriodEndingDate', 'reportingPeriodEndDate', 'originationDate',
          'vehicleManufacturerName', 'vehicleModelName', 'vehicleModelYear', 'vehicleNewUsedCode',
          'vehicleTypeCode', 'vehicleValueAmount', 'vehicleValueSourceCode', 'obligorGeographicLocation',
          'originalLoanAmount', 'obligorCreditScore']


AUTO_TRUST = re.compile(r'(?i)auto|vehicle|car[_m]|leas|motor|drive')


def list_auto_trusts(http):
    # the uploader has over a thousand ABS-EE datasets (also mortgages etc.);
    # page through all of them and keep the auto loan and lease trusts
    datasets, url, params = [], 'https://huggingface.co/api/datasets', {'author': 'DenyTranDFW', 'limit': 1000}
    while url:
        r = http.get(url, params=params, timeout=120)
        r.raise_for_status()
        datasets += r.json()
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get('Link', ''))
        url, params = (nxt.group(1) if nxt else None), None
    return [d for d in datasets if AUTO_TRUST.search(d['id'].split('/')[1])]


def download(raw_dir, filings_per_trust=3):
    http = requests.Session()
    http.headers['User-Agent'] = USER_AGENT
    datasets = list_auto_trusts(http)
    print('{} auto trusts'.format(len(datasets)), flush=True)
    for ds in datasets:
        name = ds['id'].split('/')[1]
        tree = http.get('https://huggingface.co/api/datasets/{}/tree/main?recursive=true'.format(ds['id']), timeout=60).json()
        by_filing = collections.OrderedDict()
        for f in tree:
            if f['type'] == 'file' and f['path'].endswith('.parquet') and 'exhibit' not in f['path'].lower():
                by_filing.setdefault(f['path'].split('/')[0], []).append(f)
        for filing in list(by_filing)[:filings_per_trust]:
            for f in by_filing[filing]:
                out = os.path.join(raw_dir, name, f['path'])
                if os.path.exists(out) and os.path.getsize(out) == f['size']:
                    continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                for attempt in range(3):
                    try:
                        r = http.get('https://huggingface.co/datasets/{}/resolve/main/{}'.format(ds['id'], f['path']), timeout=300)
                        r.raise_for_status()
                        open(out, 'wb').write(r.content)
                        break
                    except requests.RequestException as e:
                        print('retrying {}: {}'.format(out, e))
                        time.sleep(10)
        print(name, flush=True)


def issuer(dataset_name):
    # e.g. "CarMax_Auto_Owner_Trust_2022_1_1899236" -> "carmax",
    # "Santander_Drive_Auto_Receivables_Trust_2017_2_1705710" -> "santander"
    words = []
    for i, word in enumerate(dataset_name.lower().split('_')):
        if word in STOP_WORDS or (word == 'drive' and i > 0):
            break
        words.append(word)
    return ' '.join(words)


def column(data, name):
    # not every filing has every column (older conversions, lease vs loan schemas)
    return data[name] if name in data else pd.Series(pd.NA, index=data.index)


def load_trust(folder):
    frames = []
    for filing in sorted(os.listdir(folder)):
        for fn in os.listdir(os.path.join(folder, filing)):
            path = os.path.join(folder, filing, fn)
            # each filing also has a one-row EX-103 document file; the asset data
            # file (EX-102) is the one with an assetNumber column
            if fn.endswith('.parquet') and 'assetNumber' in pq.read_schema(path).names:
                names = pq.read_schema(path).names
                frames.append(pd.read_parquet(path, columns=[c for c in NEEDED if c in names]))
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)
    # a loan appears in every monthly filing; keep its first appearance
    # (loan filings say reportingPeriodEndingDate, lease filings reportingPeriodEndDate)
    period = column(data, 'reportingPeriodEndingDate').fillna(column(data, 'reportingPeriodEndDate'))
    data['_period'] = pd.to_datetime(period, errors='coerce')
    return data.sort_values('_period').drop_duplicates('assetNumber', keep='first')


def convert(data, dataset_name):
    cik = dataset_name.rsplit('_', 1)[-1].lstrip('0')
    orig = data['originationDate']
    if not pd.api.types.is_datetime64_any_dtype(orig):
        orig = pd.to_datetime(orig.astype(str), format='%m/%Y', errors='coerce')
    out = pd.DataFrame({
        'Price': pd.to_numeric(data['vehicleValueAmount'], errors='coerce'),
        'Make': data['vehicleManufacturerName'].astype(str).str.strip().str.title().replace(MAKES),
        'Model': data['vehicleModelName'].astype(str).str.strip().str.title(),
        'Year': pd.to_numeric(data['vehicleModelYear'], errors='coerce').astype('Int64'),
        'Body Style': pd.to_numeric(column(data, 'vehicleTypeCode'), errors='coerce').map(VEHICLE_TYPES).fillna('Other'),
        'City': '',
        'State': column(data, 'obligorGeographicLocation').fillna('').astype(str).str.strip().str.upper(),
        'Milage': pd.NA,
        'Color': '',
        'Date Scraped': orig.dt.strftime('%Y-%m-%d'),
        'Source': 'absee:' + issuer(dataset_name),
        'Listing ID': cik + ':' + data['assetNumber'].astype(str),
        'Status': pd.to_numeric(column(data, 'vehicleNewUsedCode'), errors='coerce').map(NEW_USED).fillna('Unknown'),
        'URL': 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=' + cik,
        # extra columns kept for analysis (train.py ignores them)
        'Trust': dataset_name,
        'Lease': bool(re.search(r'(?i)leas', dataset_name)),
        'Value Source Code': column(data, 'vehicleValueSourceCode'),
        'Loan Amount': pd.to_numeric(column(data, 'originalLoanAmount'), errors='coerce'),
        'Credit Score': pd.to_numeric(column(data, 'obligorCreditScore'), errors='coerce'),
    })
    out = out[out['Price'] > 0]
    return out.reindex(columns=COLUMNS + ['Trust', 'Lease', 'Value Source Code', 'Loan Amount', 'Credit Score'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--download', action='store_true', help='fetch the filings first')
    parser.add_argument('--filings', type=int, default=3, help='filings to fetch per trust (the first holds the initial pool)')
    parser.add_argument('--raw', default=RAW)
    parser.add_argument('--out', default='data/absee_vehicles.csv')
    args = parser.parse_args()

    if args.download:
        download(args.raw, args.filings)

    # trusts are converted one at a time and appended, so memory stays small
    total, by_source, by_status = 0, collections.Counter(), collections.Counter()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as out:
        for name in sorted(os.listdir(args.raw)):
            folder = os.path.join(args.raw, name)
            if not os.path.isdir(folder):
                continue
            data = load_trust(folder)
            if data is None:
                continue
            rows = convert(data, name)
            rows.to_csv(out, index=False, header=(total == 0))
            total += len(rows)
            by_source.update(rows['Source'].value_counts().to_dict())
            by_status.update(rows['Status'].value_counts().to_dict())
            print('{:<70} {:>9,} vehicles'.format(name, len(rows)), flush=True)
    print('\nsaved {:,} vehicles to {}'.format(total, args.out))
    print(dict(by_status))
    print(dict(by_source.most_common()))


if __name__ == '__main__':
    main()
