"""Fetch ONE cars.com results page, save it, and check whether the scraper can still parse it.

Makes a single request, the same one the split*.py scrapers make, so you can see
whether the site still has the page format the scrapers expect without starting a
full crawl. The page is saved as HTML so it can be inspected or shared.

Usage:
    python Code/fetch_one_page.py                      # zip 10001, page 1
    python Code/fetch_one_page.py --zip 60601 --page 2
    python Code/fetch_one_page.py --from-file page_10001_1.html   # re-check a saved page, no request
"""
import argparse
import os
import re

import requests

SCRAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split1.py')


def page_url(zip, page_num):
    # same URL as cars_call() in split*.py
    return ('https://www.cars.com/for-sale/searchresults.action/?page=' + str(page_num) +
            '&perPage=100&rd=99999&searchSource=GN_BREADCRUMB&sort=distance-nearest&zc=' + str(zip))


def scraper_functions():
    # load the scraper's functions without running its crawl loop, which starts at
    # module level and needs zip_code_database.csv
    src = open(SCRAPER).read().split('zip_codes_data = pd.read_csv')[0]
    ns = {}
    exec(compile(src, SCRAPER, 'exec'), ns)
    return ns


def parse(html):
    from bs4 import BeautifulSoup as bsoup

    ns = scraper_functions()
    # cars_call() returns the page's <script> tags as one string; do the same
    # with the saved HTML instead of downloading it again
    scripts = str(bsoup(html, features='lxml').find_all('script'))
    ns['cars_call'] = lambda zip, page_num: scripts
    return scripts, ns['organize_list_cars'](None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zip', default='10001')
    parser.add_argument('--page', type=int, default=1)
    parser.add_argument('--from-file', help='check a page saved earlier instead of fetching')
    parser.add_argument('--out', help='where to save the page (default page_<zip>_<page>.html)')
    args = parser.parse_args()

    if args.from_file:
        html = open(args.from_file, encoding='utf-8').read()
        print('read', args.from_file)
    else:
        url = page_url(args.zip, args.page)
        print('GET', url)
        response = requests.get(url, timeout=30)
        html = response.text
        out = args.out or 'page_{}_{}.html'.format(args.zip, args.page)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(html)
        print('HTTP {}, {:,} bytes, saved to {}'.format(response.status_code, len(html), out))

    title = re.search(r'<title>(.*?)</title>', html, re.S | re.I)
    print('page title:', title.group(1).strip() if title else '(none)')
    if 'Attention Required' in html or 'cf-chl' in html or 'Just a moment' in html:
        print('\nBLOCKED: cars.com\'s Cloudflare bot protection refused the request, so there is nothing to parse.')
        return

    scripts, rows = parse(html)
    print('\nwhat the scraper looks for:')
    print('  listing blocks ("type":"inventory"):', scripts.count('"type":"inventory"'))
    print('  colors:', len(re.findall('/","color":"(.*?)"},{"@context":"http://schema.org"', scripts)))
    print('  rows parsed:', len(rows))
    for row in rows[:5]:
        print('   ', row)

    if rows:
        print('\nOK: the scraper can still parse this page.')
    else:
        print('\nNO ROWS: the page format has changed since the scraper was written. '
              'Send the saved HTML file so the parsing can be updated.')


if __name__ == '__main__':
    main()
