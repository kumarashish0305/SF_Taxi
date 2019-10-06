import pickle
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import datetime as dt
import os
from sklearn.preprocessing import LabelEncoder

#Read the parquet dataframe and convert to pandas dataframe
df = sqlContext.read.parquet('/mnt/parquet/sf_taxi')
df = df.toPandas()

#Read time as time format
df['time'] = pd.to_datetime(df['time'], unit='s')

#Remove no-trip values
df = df.sort_values(by=['TaxiID', 'time']) 
# Re-index DataFrame 
i = pd.Index(np.arange(df.shape[0]))
df.index = i
 
# Consider unexpired trip as "no-trip" 
occ = df.occupancy 
if occ.values[-1] == 1: 
    df.loc[df.index[occ[occ == 0].index[-1] + 1:], "occupancy"] = 0

#Encode Taxi names to Integral ID
def encode_feature(feature, dataframe): 
  """
  Encode the labels for the given feature across both the train and test datasets. 
  """
  encoder = LabelEncoder()
  dfList = list(df[feature])
  encoder.fit(dfList)
  se = pd.Series(encoder.transform(dfList))
  # Add new column to the datasets with encoded values
  return se.values

df['TaxiID_ENCODED'] = encode_feature('TaxiID', df)

#Round off Coordinates by 4 decimals to create dataset with point to point move
df['latitude'] = round(df['latitude'], 4)
df['longitude'] = round(df['longitude'], 4)
df['col'] = df['TaxiID_ENCODED'] + df['occupancy'] + df['latitude'] + df['longitude']
df['col2'] = df['col'].diff().eq(0)

#Remove all rows where GPS position  and occupancy does not changes
df =  df[df['col2']==False]
df = df.iloc[:,0:6]

# Re-index DataFrame 
i = pd.Index(np.arange(df.shape[0]))
df.index = i

# Generate Trip ID from one trip to next trip including waiting time in between
df['occnext'] = df['occupancy'].shift(1)
df = df.fillna(0)
df['occnext'] = df['occupancy'] + df['occnext'].astype(int) * 10
df['ID'] = np.where(df['occnext'] == 10, 1, 0)
df.iloc[0,7] = 1
df['TripID'] = df['ID'].cumsum()

#Remove additional columns not needed
df = df.iloc[:,[0,1,2,3,4,5,8]]
df['coordinates'] = list(zip(df.latitude, df.longitude))

#Creatting smaller data subset for Caltrain a major Station for model development
#Filtering out trips which are not part of Caltrain by using avg lat / avg long and then using bounds (limited area in neighbourhood of Caltrain)
def remove_outliers(df):
  bounds = (37.7767, -122.400, 37.8772, -122.390)
  data = df.groupby("TripID",  as_index=False).agg({'latitude' : 'mean', 'longitude' : 'mean'})
  indices = np.where( 
        (data.latitude  >= bounds[0]) & 
        (data.longitude >= bounds[1]) &
        (data.latitude  <= bounds[2]) & 
        (data.longitude <= bounds[3])
  )
  data = data.iloc[indices]
  lst = list(data['TripID'])
  data2 = df[df['TripID'].isin(lst)]
  # Re-index DataFrame 
  i = pd.Index(np.arange(data2.shape[0]))
  data2.index = i
  return data2
  
dfinal = remove_outliers(df)

def lat_long_new(df):
  df['latnext'] = np.where(df['latitude'].shift(1).notnull(), df['latitude'].shift(1), df['latitude'])
  df['latnext'] = np.where(df['TripID'].shift(1) == df['TripID'], df['latnext'], df['latitude'])
  df['longnext'] = np.where(df['longitude'].shift(1).notnull(), df['longitude'].shift(1), df['longitude'])
  df['longnext'] = np.where(df['TripID'].shift(1) == df['TripID'], df['longnext'], df['longitude'])def lat_long_new(df):
  df['latnext'] = np.where(df['latitude'].shift(1).notnull(), df['latitude'].shift(1), df['latitude'])
  df['latnext'] = np.where(df['TripID'].shift(1) == df['TripID'], df['latnext'], df['latitude'])
  df['longnext'] = np.where(df['longitude'].shift(1).notnull(), df['longitude'].shift(1), df['longitude'])
  df['longnext'] = np.where(df['TripID'].shift(1) == df['TripID'], df['longnext'], df['longitude'])
  return df

dfinal = lat_long_new(dfinal)
  return df

dfinal = lat_long_new(dfinal)

#Calculate distance using haversine formula
def np_haversine(dataframe): 
  REarth = 6371
  latitude = dataframe['latitude']
  longitude = dataframe['longitude']
  latnext = dataframe['latnext']
  longnext = dataframe['longnext']
  lat = np.abs(latitude - latnext) * np.pi / 180
  lon = np.abs(longitude - longnext) * np.pi / 180
  lat1 = latitude * np.pi / 180
  lat2 = latnext * np.pi / 180
  a = np.sin(lat / 2) * np.sin(lat / 2) + np.cos(latitude) * np.cos(latnext) * np.sin(lon / 2) * np.sin(lon / 2)
  d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  return REarth * d

dfinal['Distance'] = np_haversine(dfinal)

#Create final dataset with unique values
def time_distance(df):
  df1 = df[df["occupancy"] == 1].groupby("TripID", as_index=False)['time'].min()
  df1.rename(columns={"TripID": "TripID", "time": "Start_time_trip"}, inplace = True)

  df2 = df.groupby("TripID", as_index=False)['time'].min()
  df2.rename(columns={"TripID": "TripID", "time": "Start_Time_initial"}, inplace = True)
  
  df3 = df.groupby("TripID", as_index=False)['time'].max()
  df3.rename(columns={"TripID": "TripID", "time": "Stop_Time"}, inplace = True)
  
  df4 = df[df["occupancy"] == 0].groupby("TripID", as_index=False)['Distance'].sum()
  df4.rename(columns={"TripID": "TripID", "Distance": "Waiting_Distance"}, inplace = True)
  
  df5 = df.groupby(["TripID","TaxiID_ENCODED"], as_index=False)['Distance'].sum()
  df5.rename(columns={"TripID": "TripID", "TaxiID_ENCODED": "TaxiID" , "Distance": "Total_Distance"}, inplace = True)
  
  df6 = df.groupby("TripID", as_index=False)['latitude'].mean()
  
  df7 = df.groupby("TripID", as_index=False)['longitude'].mean()
  
  df_merge_1 = pd.merge(df1, df2, on='TripID', how='left')
  df_merge_2 = pd.merge(df3, df4, on='TripID', how='left')
  df_merge_3 = pd.merge(df_merge_1, df_merge_2, on='TripID', how='left')
  
  df_merge_4 = pd.merge(df5, df6, on='TripID', how='left')
  df_merge_5 = pd.merge(df_merge_3, df_merge_4, on='TripID', how='left')
  
  df_merge_col = pd.merge(df_merge_5, df7, on='TripID', how='left')
  # Rename indexes
  i = pd.Index(np.arange(df_merge_col.shape[0]))
  df_merge_col.index = i
  
  #Remove rows with na
  df_merge_col.dropna()
  return df_merge_col

data = time_distance(dfinal)

#Add date based variables
def time_cal(df):
  df['Dayhour'] =  df['Stop_Time'].dt.hour
  df['Dayofweek'] = df['Stop_Time'].dt.dayofweek
  df['Month'] = df['Stop_Time'].dt.month
  df['Waiting_Time'] = (df['Start_time_trip'] - df['Start_Time_initial']).dt.seconds
  df['Total_Time'] = (df['Stop_Time'] - df['Start_Time_initial']).dt.seconds
  return df

data = time_cal(data)

#Make a directory for parquet format
dbutils.fs.mkdirs("/mnt/parquet2")
# Create a Spark DataFrame from a pandas DataFrame using Arrow
sdf = spark.createDataFrame(data)
sdf.write.mode("overwrite").format("parquet").save("/mnt/parquet2/sf_taxi_new")
