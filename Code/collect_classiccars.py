"""Collect ClassicCars.com listings.

ClassicCars.com's terms don't ban automated access, as long as it doesn't interfere
with the site and doesn't disguise who is making the requests; its robots.txt allows
the listing pages. So this collector identifies itself honestly, obeys robots.txt,
waits between requests and stops if the site starts refusing them.

Two passes, both resumable (progress is saved to the output CSV):
  1. search pages (15 cars each): price, year, make, model, location
  2. --details: each listing's page for odometer, colors, transmission, engine,
     trim, VIN and title status (one request per car, so it takes much longer)

Usage:
    python Code/collect_classiccars.py                    # all search pages
    python Code/collect_classiccars.py --details          # then fill in details
    python Code/collect_classiccars.py --details --max-details 500
"""
import argparse
import html
import json
import re
import time
from datetime import date
from urllib.robotparser import RobotFileParser

from collect_common import USER_AGENT, enrich_with_vins, read_rows, session, write_rows

SITE = 'https://classiccars.com'
STATES = {'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
          'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
          'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
          'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD', 'massachusetts': 'MA',
          'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT',
          'nebraska': 'NE', 'nevada': 'NV', 'new-hampshire': 'NH', 'new-jersey': 'NJ', 'new-mexico': 'NM',
          'new-york': 'NY', 'north-carolina': 'NC', 'north-dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
          'oregon': 'OR', 'pennsylvania': 'PA', 'rhode-island': 'RI', 'south-carolina': 'SC',
          'south-dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT',
          'virginia': 'VA', 'washington': 'WA', 'west-virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY',
          'district-of-columbia': 'DC'}
# label on the listing page -> output column
DETAILS = {'Odometer': 'Milage', 'Exterior Color': 'Color', 'Interior Color': 'Interior Color',
           'Transmission': 'Transmission', 'Engine Size': 'Engine', 'Trim Level': 'Trim',
           'VIN': 'VIN', 'Title Status': 'Title Status'}
LABELS = ['Listing ID', 'Price', 'Location', 'Year', 'Make', 'Model', 'Exterior Color', 'Interior Color',
          'Transmission', 'Engine Size', 'Odometer', 'Stock Number', 'Trim Level', 'VIN', 'Title Status',
          'Listed By', 'Get Collector Car Insurance']


class Blocked(Exception):
    pass


class Crawler:
    def __init__(self, delay):
        self.http = session()
        self.delay = delay
        # fetch robots.txt with our own User-Agent: RobotFileParser.read() uses
        # urllib's default one, which the site refuses (and a refusal reads as "disallow all")
        self.robots = RobotFileParser()
        self.robots.parse(self.http.get(SITE + '/robots.txt', timeout=60).text.splitlines())

    def get(self, path):
        url = SITE + path
        if not self.robots.can_fetch(USER_AGENT, url):
            raise Blocked('robots.txt disallows ' + url)
        time.sleep(self.delay)
        r = self.http.get(url, timeout=60)
        if r.status_code in (403, 429, 503):
            raise Blocked('HTTP {} for {}'.format(r.status_code, url))
        r.raise_for_status()
        return r.text


def location_from_url(url):
    # /listings/view/2117409/1968-datsun-2000-for-sale-in-grass-valley-california-95945
    m = re.search(r'-for-sale-in-(.+?)-(\d{5})$', url or '')
    if not m:
        return '', ''
    place = m.group(1)
    for name in sorted(STATES, key=len, reverse=True):
        if place.endswith('-' + name):
            return place[:-len(name) - 1].replace('-', ' ').title(), STATES[name]
    return place.replace('-', ' ').title(), ''


def parse_search_page(page, today):
    rows = []
    for block in re.findall(r'(?is)<script[^>]+application/ld\+json[^>]*>(.*?)</script>', page):
        try:
            item = json.loads(block)
        except ValueError:
            continue
        if str(item.get('@type', '')).lower() != 'car':
            continue
        offer = item.get('offers') or {}
        price = offer.get('price') or ''
        city, state = location_from_url(offer.get('url'))
        rows.append({
            'Price': price if str(price) not in ('0', '0.0') else '',
            'Make': item.get('manufacturer') or item.get('brand') or '',
            'Model': item.get('model') or '',
            'Year': item.get('modelDate') or '',
            'City': city, 'State': state,
            'Date Scraped': today, 'Source': 'classiccars',
            'Listing ID': item.get('sku') or '',
            'Status': 'Active',
            'URL': SITE + offer.get('url', ''),
        })
    return [row for row in rows if row['Listing ID']]


def parse_detail_page(page):
    text = re.sub(r'\s+', ' ', html.unescape(re.sub(r'(?s)<(script|style).*?</\1>|<[^>]+>', ' ', page)))
    start = text.find('Vehicle Details')
    text = text[start:start + 3000] if start >= 0 else ''
    found = {}
    labels = '|'.join(re.escape(l) for l in LABELS)
    for label, column in DETAILS.items():
        m = re.search(r'%s:\s*(.*?)\s*(?=(?:%s):|$)' % (re.escape(label), labels), text)
        if m and m.group(1):
            value = m.group(1).strip()
            if column == 'Milage':
                value = re.sub(r'[^\d]', '', value)
            found[column] = value
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='data/classiccars_vehicles.csv')
    parser.add_argument('--delay', type=float, default=2.0, help='seconds between requests (default 2)')
    parser.add_argument('--max-pages', type=int, default=2000)
    parser.add_argument('--skip-search', action='store_true', help='only run the details pass')
    parser.add_argument('--details', action='store_true', help='also fetch each listing page')
    parser.add_argument('--max-details', type=int, default=0, help='stop the details pass after this many (0 = all)')
    args = parser.parse_args()

    today = date.today().isoformat()
    crawler = Crawler(args.delay)
    rows = {row['Listing ID']: row for row in read_rows(args.out)}
    print('{:,} listings already in {}'.format(len(rows), args.out))

    def save():
        write_rows(args.out, rows.values())

    try:
        if not args.skip_search:
            seen_today, empty_pages = set(), 0
            for page_num in range(1, args.max_pages + 1):
                found = parse_search_page(crawler.get('/listings/find?p={}'.format(page_num)), today)
                new = [row for row in found if row['Listing ID'] not in seen_today]
                for row in new:
                    seen_today.add(row['Listing ID'])
                    previous = rows.get(row['Listing ID'])
                    if previous:   # keep the first-seen date and anything from the details pass
                        row = {**previous, **{k: v for k, v in row.items() if v}, 'Date Scraped': previous['Date Scraped']}
                    rows[row['Listing ID']] = row
                empty_pages = 0 if new else empty_pages + 1
                if page_num % 25 == 0:
                    save()
                    print('page {}: {:,} listings seen this run, {:,} total'.format(page_num, len(seen_today), len(rows)), flush=True)
                if empty_pages >= 3:
                    break
            save()
            print('search pass done: {:,} listings seen this run'.format(len(seen_today)))

        if args.details:
            todo = [row for row in rows.values() if not row.get('Milage') and not row.get('Title Status')]
            if args.max_details:
                todo = todo[:args.max_details]
            print('details pass: {:,} listings to fetch'.format(len(todo)), flush=True)
            for i, row in enumerate(todo, 1):
                details = parse_detail_page(crawler.get(row['URL'][len(SITE):]))
                # mark as visited even when the page has no odometer
                row.update(details or {'Title Status': 'n/a'})
                if i % 50 == 0:
                    save()
                    print('details: {:,}/{:,}'.format(i, len(todo)), flush=True)
    except Blocked as e:
        print('stopping: {}'.format(e))
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        save()

    todo = [row for row in rows.values() if row.get('VIN') and not row.get('Body Style')]
    if todo:
        print('decoded {:,} VINs with NHTSA vPIC'.format(enrich_with_vins(todo)))
        save()
    with_miles = sum(1 for row in rows.values() if row.get('Milage'))
    print('saved {:,} listings to {} ({:,} with mileage)'.format(len(rows), args.out, with_miles))


if __name__ == '__main__':
    main()
