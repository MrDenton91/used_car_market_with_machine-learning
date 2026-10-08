from bs4 import BeautifulSoup as bsoup
import requests
import time
import copy
import pandas as pd
import re
import csv
from datetime import date
import numpy as np
import os

#from bson.objectid import ObjectId

# connect to the hosted MongoDB instance

#Load webpage content 
## now I can feed my method a zipcode and a page number
def cars_call(zip,page_num):
    ##this is the original page
    #page = requests.get('https://www.cars.com/for-sale/searchresults.action/?page='+str(page_num)+'&perPage=100&rd=20&searchSource=PAGINATION&sort=relevance&zc='+str(zip))
    
    #This has all the cars sorted by there distacne from a given zip code
    page = requests.get('https://www.cars.com/for-sale/searchresults.action/?page='+str(page_num)+'&perPage=100&rd=99999&searchSource=GN_BREADCRUMB&sort=distance-nearest&zc='+str(zip))
    
    #convert to a beautiful soup object:
    soup = bsoup(page.content, features="lxml")
    #show contents
    soup.prettify()

    # auto tempest isn't working.. did they somehow make everything private? D:
    #start scraping find and find_all
    body = soup.find_all('script')
    
    #body = soup.find('class')
    #body = soup.find('div' )
    # yes or no answer if it is certified pre-owned
    #spec = body.find_all('span')
    return str(body)

#this moster of method goes through a each page scrapping what I want
def organize_list_cars(zip, page_num):
    delimeters = '"type"'
    # fetch the page once and parse both the listings and the colors out of it
    strings = cars_call(zip, page_num)
    lstin = re.split(delimeters, strings)

    color1 = re.findall('/","color":"(.*?)"},{"@context":"http://schema.org"', strings)

    #shorttening list of stuff :0
    new_list = []
    for i in lstin:
        if i.startswith('' ':"inventory"') == True:
            new_list.append(i)

    # one regex per feature, in the column order of the csv:
    # Price, Make, Model, Year, Body Style, City, State, Milage (Color comes from color1)
    patterns = [',"price":(.+),"mileage":',
                '"make":"(.+)","makeId"',
                '"model":"(.+)","modelId"',
                '"year":(.+),"trim"',
                '"bodyStyle":"(.+)","customerId"',
                '"city":"(.+),"state":',
                ',"state":"(.+)","truncatedDescription',
                ',"mileage":(.+),"vin":']

    #populate one row of feature values per car
    container = []
    for i, car in enumerate(new_list):
        row = []
        for pattern in patterns:
            found = re.findall(pattern, car)
            row.append(found[0].replace('"', '').strip() if found else '')
        row.append(color1[i] if i < len(color1) else '')

        # skip cars missing a feature and duplicate listings
        if all(row) and row not in container:
            container.append(row)
    return container


#I need a way to write all this information to a csv file for manipulation and EDA
def populate_car_list(zip,page_num):
    # csv.writer quotes values that contain commas (e.g. "Washington, D.C.")
    with open('./carlist6.csv','a', newline='') as f:
        writer = csv.writer(f)
        # last column: the date this row was scraped / added to the data set
        scraped = date.today().isoformat()
        for item in organize_list_cars(zip,page_num):
            writer.writerow(item + [scraped])

# I already created a list of all zip code within the United states
# I just need to import it
zip_codes_data = pd.read_csv('zip_code_database.csv')
zips = zip_codes_data['zip'].to_numpy()

#This for loop should start my data collection 
for zi in zips:
    #there are bad zip code before getting to the first one in New York
    #So about 32,000 zip codes in total there's going to be cars per page
    if zi >= 32500 and zi < 37000:
    # at most I'm only allowed to see 50 pages beacuse of the cars.com website, which covers about 11 miles.

            for a in range(1,50):
                try:
                    populate_car_list(zi,a)
                except:
                    pass  
        
### 32,000*5,000 = 160,000,000 cars!
## but I'm expected a 98% duplication, leaving me 3.2 million cars
## considering it's been reported that there's about 40 million cars being sold in the US alone, I should be okay.

# This is mostly for myself :D 
