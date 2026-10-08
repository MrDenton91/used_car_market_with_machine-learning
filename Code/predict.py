"""Price a single car with a model saved by train.py.

Usage:
    python Code/predict.py --make Toyota --model Camry --year 2016 --mileage 45000 \
        --body-style Sedan --city Denver --state CO --color Silver
"""
import argparse

import joblib
import pandas as pd


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
    args = parser.parse_args()

    car = pd.DataFrame([{
        'Make': args.make, 'Model': args.model, 'Body Style': args.body_style,
        'City': args.city, 'State': args.state, 'Color': args.color,
        'Age': args.scrape_year - args.year, 'Milage': args.mileage,
    }])
    price = joblib.load(args.model_file).predict(car)[0]
    print('${:,.0f}'.format(price))


if __name__ == '__main__':
    main()
