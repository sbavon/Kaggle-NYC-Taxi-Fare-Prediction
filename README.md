# Kaggle NYC Taxi Fare Prediction Kaggle Solution (Top 2% Ranked 21st/1400)

This repository is the solution I used to obtain the top 2% ranking of NYC Taxi Fare Prediction competition.
It can be separated into three parts: data cleaning, data preprocessing, and training

## Data cleaning ( refer to `data_cleaning.py`)
* remove null records
* remove records whose locations are not within range provided in test data
* remove data points in sea
* eliminate outlier according to fare distribution

## Data preprocessing ( refer to `Data_preprocessing.py`)
* the new feature `cluster` is added. 
	* during data exploration, I found that the fare/distance ratio is varying based on the location. So, I add the new categorical feature to specify area code
	* I used **HDBScan** technique to obtain the clustering model. Then, I use this model to predict area code of the data.
		<< image area>>
* the new feature `distance` is added
* the new feature `distance to airport` is added since I observe that the price is significantly connect with the distance from the airport

## train and predict ( refer to `train_predict.py`)
* **lightgbm** is used, and it was trained in the Amazon EC2 instance
* data type of categorical data is changed to float32 to prevent memory surge due to Lightgbm python package. 
(The library will convert all data to float. So if the data is integer, the library will create a new data in float, which contains much memory space.)