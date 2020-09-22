from bs4 import BeautifulSoup as bsoup
import requests
import time
import copy
import pandas as pd
import re
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
    strings = cars_call(zip, page_num)
    lstin = re.split(delimeters, strings)
    color = []
    
    color.append(((re.findall('/","color":"(.*?)"},{"@context":"http://schema.org"',cars_call(zip, page_num) ))))
    color1 = color[0]
    
    #shorttening list of stuff :0
    new_list = []
    for i in lstin:
        if i.startswith('' ':"inventory"') == True:
            new_list.append(i)
    ##i'm just creating features lists for each charateristic i'm intrested in.
    price = []
    make =[]
    model = []
    year = []
    bodyStyle = []
    sellerRating = []
    city = []
    state = []
    mileage = []
    rating = []
    
    #populate feature lists values
    for i in new_list:
        make.append(re.findall('"make":"(.+)","makeId"', i))
        model.append((re.findall('"model":"(.+)","modelId"', i)))
        year.append((re.findall('"year":(.+),"trim"', i)))
        bodyStyle.append((re.findall('"bodyStyle":"(.+)","customerId"', i)))
        sellerRating.append((re.findall(',"rating":(.+),"reviewCount"', i)))
        city.append((re.findall('"city":"(.+),"state":', i)))
        price.append((re.findall(',"price":(.+),"mileage":', i)))
        mileage.append((re.findall(',"mileage":(.+),"vin":', i)))
        state.append((re.findall(',"state":"(.+)","truncatedDescription', i)))
        color.append(((re.findall('","color":"(.+)"},{"@context":"', i))))
        rating.append(re.findall('"rating":(.+),"review',i))
        
        
    #creaing an uncleaning master list of cars
    masterlist = []
    for i in range(len(make)):
        #appending everything to master list before cleaning, I broke it up into two so it's a bit easier to read.
        try:
            masterlist.append(str(price[i]) + str(make[i]) + str(model[i]) + str(year[i]) + str(bodyStyle[i]) +str(city[i])+ str(state[i])+ str(mileage[i]) +',' +str(color1[i])  )
        except:
            continue
        #masterlist.append(str(seller_label[i]) + str(rating[i]) + str(review_count[i]) + str(distance_from_zip[i]))
    #cleaning the master list 
    
    container = []
    for i in range(len(masterlist)):
        cars = masterlist[i]
        cars = cars.replace('[]', ',')
        cars = cars.replace('[', '')
        cars = cars.replace(']', '')
        cars = cars.replace("''",',')
        cars = cars.replace("'", '')
        cars = cars.replace('"','')
        if cars.count(',') == 8 and cars not in container:
            container.append(cars)
        else:
            pass
    return container


#I need a way to write all this information to a csv file for manipulation and EDA
def populate_car_list(zip,page_num):
    with open('./carlist6.csv','a') as f:
        for item in organize_list_cars(zip,page_num):
            f.write("%s\n" % item)
        
    pass
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
