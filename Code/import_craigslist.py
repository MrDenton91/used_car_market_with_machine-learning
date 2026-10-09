"""Convert the open "Used Cars Dataset" (US Craigslist listings, 2021) to the collectors' CSV format.

The dataset is published under CC0 on OpenML and CC-BY-4.0 on Zenodo:
https://zenodo.org/records/10457828 (download "Used Cars Dataset.zip", ~275 MB).

Usage:
    python Code/import_craigslist.py data/raw/craigslist_used_cars.zip
"""
import argparse
import zipfile

import pandas as pd

from collect_common import COLUMNS

RENAME = {
    'price': 'Price', 'manufacturer': 'Make', 'model': 'Model', 'year': 'Year', 'type': 'Body Style',
    'region': 'City', 'state': 'State', 'odometer': 'Milage', 'paint_color': 'Color',
    'posting_date': 'Date Scraped', 'id': 'Listing ID', 'VIN': 'VIN', 'transmission': 'Transmission',
    'cylinders': 'Engine', 'drive': 'Drive Type', 'fuel': 'Fuel Type', 'title_status': 'Title Status',
    'url': 'URL',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help='the downloaded zip, or vehicles.csv')
    parser.add_argument('--out', default='data/craigslist_vehicles.csv')
    args = parser.parse_args()

    if args.source.endswith('.zip'):
        with zipfile.ZipFile(args.source) as z:
            with z.open('vehicles.csv') as f:
                data = pd.read_csv(f, usecols=list(RENAME), low_memory=False)
    else:
        data = pd.read_csv(args.source, usecols=list(RENAME), low_memory=False)

    data = data.rename(columns=RENAME)
    data['Source'] = 'craigslist'
    data['Status'] = 'Listed'
    data['Make'] = data['Make'].str.title()
    data['Model'] = data['Model'].str.strip().str.title()
    data['State'] = data['State'].str.upper()
    data['City'] = data['City'].str.title()
    for col in ['Body Style', 'Color']:
        data[col] = data[col].str.title()
    data['Date Scraped'] = pd.to_datetime(data['Date Scraped'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
    for col in ['Year', 'Milage']:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Int64')

    data.reindex(columns=COLUMNS).to_csv(args.out, index=False)
    print('saved {:,} listings to {}'.format(len(data), args.out))
    print('posted {} to {}'.format(data['Date Scraped'].min(), data['Date Scraped'].max()))


if __name__ == '__main__':
    main()
