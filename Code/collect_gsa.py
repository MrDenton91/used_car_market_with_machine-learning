"""Collect cars and trucks from GSA Auctions (US government surplus) and track their final bids.

The GSA Auctions API is public and only lists open auctions, so run this daily:
every run records each vehicle's current high bid, and once an auction's end date
has passed the last bid seen is kept as its sale price (Status = Closed).

Usage:
    python Code/collect_gsa.py                       # writes data/gsa_vehicles.csv
    python Code/collect_gsa.py --api-key YOUR_KEY    # free key from api.data.gov (DEMO_KEY is rate limited)
"""
import argparse
import html
import re
from datetime import date

from collect_common import enrich_with_vins, read_rows, session, write_rows

API = 'https://api.gsa.gov/assets/gsaauctions/v2/auctions'
MAKES = ['Acura', 'Audi', 'BMW', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'Ford', 'GMC',
         'Honda', 'Hyundai', 'Infiniti', 'Jeep', 'Kia', 'Lexus', 'Lincoln', 'Mazda', 'Mercedes-Benz',
         'Mercury', 'Mitsubishi', 'Nissan', 'Ram', 'Subaru', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo',
         'Freightliner', 'International', 'Isuzu', 'Hino', 'Polaris']
ALIASES = {'chevy': 'Chevrolet', 'mercedes': 'Mercedes-Benz', 'vw': 'Volkswagen'}
NAME = re.compile(r'\b(19[5-9]\d|20[0-3]\d)\s+(%s)\b\s*(.*)' % '|'.join(
    [re.escape(m) for m in MAKES] + list(ALIASES)), re.I)


def clean_text(markup):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', markup or ''))).strip()


def parse_lot(lot, today):
    match = NAME.search(lot.get('itemName') or '')
    if not match:
        return None
    year, make, rest = match.groups()
    make = ALIASES.get(make.lower(), next(m for m in MAKES if m.lower() == make.lower()))
    # model = first words of the title, before condition notes like "- RUST ON FRAME"
    model = re.split(r'\s+-\s+|,|\(', rest)[0].strip().title() or 'Unknown'

    info = clean_text(lot.get('lotInfo'))
    odometer = re.search(r'(?i)(?:odometer|mileage|miles)\D{0,25}([\d,]{2,9})', info)
    vin = re.search(r'\b[A-HJ-NPR-Z0-9]{17}\b', info)
    color = re.search(r'(?i)\b(?:exterior\s+)?colou?r\s*[:\-]\s*([A-Za-z]+)', info)

    closed = (lot.get('aucEndDt') or '9999') < today
    return {
        'Price': lot.get('highBidAmount') or '',
        'Make': make, 'Model': model, 'Year': year,
        'City': (lot.get('propertyCity') or '').strip().title(),
        'State': (lot.get('propertyState') or '').strip(),
        'Milage': odometer.group(1).replace(',', '') if odometer else '',
        'Color': color.group(1).title() if color else '',
        'Date Scraped': today, 'Source': 'gsa',
        'Listing ID': '{}-{}'.format(lot.get('saleNo'), lot.get('lotNo')),
        'VIN': vin.group(0) if vin else '',
        'Status': 'Closed' if closed else (lot.get('auctionStatus') or '').title(),
        'Auction End': lot.get('aucEndDt') or '',
        'URL': lot.get('itemDescURL') or '',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--api-key', default='DEMO_KEY')
    parser.add_argument('--out', default='data/gsa_vehicles.csv')
    args = parser.parse_args()

    today = date.today().isoformat()
    http = session()
    r = http.get(API, params={'api_key': args.api_key, 'format': 'JSON'}, timeout=120)
    r.raise_for_status()
    lots = r.json()['Results']
    current = {row['Listing ID']: row for row in filter(None, (parse_lot(l, today) for l in lots))}
    print('{:,} open lots, {:,} cars and trucks'.format(len(lots), len(current)))

    # merge with earlier runs: keep the latest bid seen for every lot, and mark
    # lots that have dropped off the API (or passed their end date) as closed
    rows = {row['Listing ID']: row for row in read_rows(args.out)}
    for lot_id, row in rows.items():
        if lot_id not in current and row.get('Auction End', '9999') < today:
            row['Status'] = 'Closed'
    for lot_id, row in current.items():
        previous = rows.get(lot_id)
        if previous:
            # keep fields decoded earlier and the first-seen date
            row['Date Scraped'] = previous.get('Date Scraped') or today
            for key in ['Body Style', 'Drive Type', 'Fuel Type', 'Engine']:
                row[key] = row.get(key) or previous.get(key, '')
            if not row['Price']:
                row['Price'] = previous.get('Price', '')
        rows[lot_id] = row

    todo = [row for row in rows.values() if row.get('VIN') and not row.get('Body Style')]
    print('decoded {:,} VINs with NHTSA vPIC'.format(enrich_with_vins(todo, http)))

    write_rows(args.out, rows.values())
    closed = sum(1 for row in rows.values() if row.get('Status') == 'Closed')
    with_bid = sum(1 for row in rows.values() if row.get('Price'))
    print('saved {:,} vehicles to {} ({:,} with a bid, {:,} closed)'.format(len(rows), args.out, with_bid, closed))


if __name__ == '__main__':
    main()
