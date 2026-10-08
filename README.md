# Used Car Market Using Machine Learning

## Abstract
Cars.com (Cars) has a huge selection of user and dealership submitted cars for sale, this as well as its ease of scraping made it the ideal candidate for data collection.
Goal of this Machine Learning Project is simply:
* Train a model to predict the price of a car.

## Results
* a number of models predicted the price of car pretty well on unseen data compared to a baseline that predicts the median price for every car. Gradient Boosting yielded the best results with an RMSE of 8,396. RMSE of the model base is 19,298.

## Data
Cars was scraped using 20 scraper programs running in parallel and stored locally on a M.2 Form Factor. 
Each datapoint represents a unique car being sold in the U.S. Cars.com refreshes the cars being shown on their website every couple of days. As such new data could be collected and added to total data set.

## EDA
In 2019 the used car market saw 40 million transactions, according to search results, as such the data being collected needed to be compared to the known distributions for the market. See below for graphs on data

* Make Distribution on the top 10

![](Images/make_dist.png)




* Model Distribution on the top 10

![](Images/model_dis.png)




* Color Distribution on the top 10

![](Images/color_dist.png)


## Model
The general idea is to build a model that returns a price for a car given input feature information. Data was tested on a few different types of machine learning models. Gradient Boosting performed the best when compared to the baseline RMSE (average). Neural network is still a work in progress. 

### Output
The model outputs a price in USD. The raw scrape contained listings priced anywhere from $1,000 to $100,000,000; `Code/train.py` drops listings outside $500 - $500,000 (along with mileage over 500,000) as data-entry errors.

### Features
* Year
* Make
* Model
* Mileage
* City
* State
* Body Style
* Color

## Usage
```bash
pip install -r requirements.txt

# train on the scraped CSVs (headerless scraper output or combined CSVs with a header)
python Code/train.py carlist*.csv --scrape-year 2020

# price a car with the saved pipeline
python Code/predict.py --make Toyota --model Camry --year 2016 --mileage 45000 \
    --body-style Sedan --city Denver --state CO --color Silver
```

`train.py` compares a median baseline, ridge regression and gradient boosting, reports RMSE, MAE and R² on a
held-out test set, and saves the best model to `models/price_model.joblib` with its scores in
`models/price_model_metrics.json`. The saved file is a full scikit-learn pipeline (encoding + model), so it
predicts straight from raw values; only load model files you trust, since joblib/pickle files can run code
when loaded.

The scraped CSVs and `zip_code_database.csv` (used by the scrapers) are not included in this repository.
