


import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import hdbscan
from mpl_toolkits.basemap import Basemap

### function that get a clusterer, using HDBScan technique
def add_cluster(df):
  
    ### load pre-trained clusterer
    with open('clusterer.pkl', 'rb') as input:
      clusterer = pickle.load(input)
    
    ### predict cluster
    pickup_area = hdbscan.approximate_predict(clusterer, np.radians(df[['pickup_latitude', 'pickup_longitude']].values))[0]
    dropoff_area = hdbscan.approximate_predict(clusterer, np.radians(df[['dropoff_latitude', 'dropoff_longitude']].values))[0]
      
    df['pickup_area'] = pickup_area
    df['dropoff_area'] = dropoff_area

    del pickup_area
    del dropoff_area
  
    return df
  
### function that calcuates distance between two locations
def getDistance(lat1,lon1,lat2,lon2):
    r = 6373 # earth's radius
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = r*c
    
    return distance

### function that calculates distance between airports and pickup location and dropoff location
def add_extra_dist(df):

    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    
    pickup_lat = df['pickup_latitude']
    dropoff_lat = df['dropoff_latitude']
    pickup_lon = df['pickup_longitude']
    dropoff_lon = df['dropoff_longitude']
    
    df['pickup_jfk'] = getDistance(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    df['dropoff_jfk'] = getDistance(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    df['pickup_ewr'] = getDistance(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    df['dropoff_ewr'] = getDistance(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    df['pickup_lga'] = getDistance(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    df['dropoff_lga'] = getDistance(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)
    
    return df

### function that calculates distance between pickup location and dropoff location
def add_dist(df):
    df['distance'] = getDistance(df.pickup_latitude, df.pickup_longitude, 
                                      df.dropoff_latitude, df.dropoff_longitude)
    return df

### function that splits datetime into categorical data
def add_datetime(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format="%Y-%m-%d %H:%M:%S UTC")
    df['year'] = df.pickup_datetime.dt.year
    df['month'] = df.pickup_datetime.dt.month
    df['day'] = df.pickup_datetime.dt.day
    df['hour'] = df.pickup_datetime.dt.hour
    df['dayOfWeek'] = df.pickup_datetime.dt.dayofweek

### function that convert latitudes and longtitudes to radians format
def convert_to_radians(df):
    df['pickup_latitude'] = np.deg2rad(df['pickup_latitude'].values)
    df['pickup_longitude'] = np.deg2rad(df['pickup_longitude'].values)
    df['dropoff_latitude'] = np.deg2rad(df['dropoff_latitude'].values)
    df['dropoff_longitude'] = np.deg2rad(df['dropoff_longitude'].values)
    return df
  
### function that preprocess data before training the model
def preprocess_data(df):
    
    df = add_cluster(df)
    df = add_extra_dist(df)
    df = add_dist(df)
    df = add_datetime(df)
    df = convert_to_radians(df)
    
    if 'fare_amount' in df:   
      y = df['fare_amount'] # train data: keep labels
      df.drop(columns=['pickup_datetime','key','fare_amount'], inplace=True)
    else:
      y = 0 # test data: do not care about labels
      df.drop(columns=['pickup_datetime','key'], inplace=True)
    
    return df, y

