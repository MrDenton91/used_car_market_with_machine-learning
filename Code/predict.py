"""Price a single car with a model saved by train.py.

Usage:
    python Code/predict.py --make Toyota --model Camry --year 2016 --mileage 45000 \
        --body-style Sedan --city Denver --state CO --color Silver
"""
import argparse

import joblib
import pandas as pd

import pricing_model  # noqa: F401  (needed to unpickle the model)


def text_flags(description, private_seller=False):
    """The same listing-text features import_craigslist.py derives; 'Unknown' without a description."""
    from import_craigslist import TEXT_FLAGS
    import re
    if not description:
        flags = {flag: 'Unknown' for flag in TEXT_FLAGS}
        flags['Description Length'] = float('nan')
    else:
        text = description.lower()
        flags = {flag: 'Yes' if re.search(pattern, text) else 'No' for flag, pattern in TEXT_FLAGS.items()}
        flags['Description Length'] = float(len(text))
    if private_seller:
        flags['Dealer Listing'] = 'No'
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-file', default='models/price_model.joblib')
    parser.add_argument('--scrape-year', type=int, default=2020,
                        help='must match the --scrape-year the model was trained with')
    parser.add_argument('--make', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--mileage', type=float, required=True)
    parser.add_argument('--body-style', required=True)
    parser.add_argument('--city', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--color', required=True)
    parser.add_argument('--source', default='cars.com', help='market to price in: cars.com, craigslist, classiccars, gsa, absee:carmax, ...')
    parser.add_argument('--sale-year', type=int, help='year of the sale being priced (default: the --scrape-year)')
    parser.add_argument('--sale-month', type=int, default=6, help='month of the sale being priced (1-12)')
    for name in ['trim', 'transmission', 'engine', 'drive-type', 'fuel-type', 'title-status']:
        parser.add_argument('--' + name, default='Unknown')
    for name in ['displacement', 'horsepower', 'doors']:
        parser.add_argument('--' + name, type=float, default=float('nan'), help='engine litres / hp / door count, if known')
    parser.add_argument('--condition', default='Unknown', help="seller's rating: New, Like New, Excellent, Good, Fair, Salvage")
    parser.add_argument('--description', default='', help="the listing text; the same Yes/No flags as training are read from it")
    parser.add_argument('--private-seller', action='store_true', help='not a dealer listing (otherwise read from the description)')
    args = parser.parse_args()

    sale_year = args.sale_year or args.scrape_year
    car = pd.DataFrame([{
        'Source': args.source, 'Make': args.make.title(),
        'Model': ''.join(ch for ch in args.model.upper() if ch.isalnum()) or 'UNKNOWN',
        'Body Style': args.body_style, 'City': args.city, 'State': args.state, 'Color': args.color,
        'Trim': args.trim, 'Transmission': args.transmission, 'Engine': args.engine,
        'Drive Type': args.drive_type, 'Fuel Type': args.fuel_type, 'Title Status': args.title_status,
        'Age': sale_year - args.year, 'Sale Year': sale_year, 'Sale Time': sale_year + (args.sale_month - 1) / 12,
        'Milage': args.mileage, 'Displacement': args.displacement, 'Horsepower': args.horsepower, 'Doors': args.doors,
        'Condition': args.condition, **text_flags(args.description, args.private_seller),
    }])
    price = joblib.load(args.model_file).predict(car)[0]
    print('${:,.0f}'.format(price))


if __name__ == '__main__':
    main()
