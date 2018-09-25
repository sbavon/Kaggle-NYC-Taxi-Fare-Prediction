
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### function that remove data point on water
### This function is from Albert van Breemen
### https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx*(longitude - BB[0])/(BB[1]-BB[0])).astype('int'),                (dy - dy*(latitude - BB[2])/(BB[3]-BB[2])).astype('int')

    # define bounding box
    BB = (-74.5, -72.8, 40.5, 41.8)
    
    # read nyc mask and turn into boolean map with
    # land = True, water = False
    nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
    
    # calculate for each lon,lat coordinate the xy coordinate in the mask map
    pickup_x, pickup_y = lonlat_to_xy(df.pickup_longitude, df.pickup_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)
    dropoff_x, dropoff_y = lonlat_to_xy(df.dropoff_longitude, df.dropoff_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)    
    # calculate boolean index
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    
    # return only datapoints on land
    return df[idx]

### function used for data cleaning
def clean_data(df):
  
  ### remove records that contain NaN value
  df = df.dropna(how = 'any', axis = 'rows')
  
  ### remove records that contain longitude outside of the range of test data
  df = df[(df['pickup_longitude'] <= -72.8)
         & (df['pickup_longitude'] >= -74.5)]
  df = df[(df['dropoff_longitude'] <= -72.8)
       & (df['dropoff_longitude'] >= -74.5)]
  df = df[(df['pickup_latitude'] <= 41.8)
       & (df['pickup_latitude'] >= 40.5)]
  df = df[(df['dropoff_latitude'] <= 41.8)
       & (df['dropoff_latitude'] >= 40.5)]
  
  ### remove records that their fare amount are equal to 0 or exceed 200
  df = df[(df['fare_amount'] > 0) & (df['fare_amount'] <= 200)]
  
  ### remove records that have location on the sea
  df = remove_datapoints_from_water(df)
  
  return df

### function that clean the whole data and save the result in current directory
def clean_whole_data(infilename, outfilename, chunksize):
  
  iteration = 0
  for data in pd.read_csv(infilename, chunksize = chunksize):
    x = clean_data(data)
    if iteration == 0:
      x.to_csv(outfilename,
            index=False,
            header=True,
            mode='w')#size of data to append for each loop
    else:
      x.to_csv("outfilename",
            index=False,
            header=False,
            mode='a')#size of data to append for each loop
    iteration += 1
    print(iteration)

def main():
  infilename = 'train.csv'
  outfilename = 'cleaned_train.csv'
  chunksize = 100000
  clean_whole_data(infilename, outfilename, chunksize)

