import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from matplotlib.colors import LogNorm 
from matplotlib import gridspec
import calendar
from matplotlib.ticker import  MultipleLocator, FormatStrFormatter

#Command Cell 2
#Read the parquet dataframe and convert to pandas dataframe
df = sqlContext.read.parquet('/mnt/parquet2/sf_taxi_new')
df = df.toPandas()

df['latitude'] = round(df['latitude'], 4)
df['longitude'] = round(df['longitude'], 4)
df['coordinates'] = list(zip(df.latitude, df.longitude))

#Command Cell 3
#Count of Trips by Day of the Week
plt.figure(figsize=(7.5,4))
sns.countplot(df['Dayofweek'])
plt.gca().set_xticklabels(calendar.day_name)
plt.xticks(fontsize=8)
plt.xlabel('Day of the week')
plt.show()
display()

#Count of Trips by Day Hour
#Max trips at 16:00 hour or afternoon to or from Caltrain indicating peak traffic time
plt.figure(figsize=(7.5,4))
sns.countplot(df['Dayhour'])
plt.gca().xaxis.set_major_locator(MultipleLocator(4))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.xticks(fontsize=8)
plt.xlabel('Day Hour')
plt.show()
display()


#Durations of trips
#Mean trip duration is 7 mins. for Caltrain Location.
plt.figure(figsize=(7.5,4))
bins = np.arange(60, df.Total_Time.max(), 60)
binned = pd.cut(df.Total_Time, bins, labels=bins[:-1]/60, include_lowest=True)
sns.countplot(binned, color='royalblue')
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.xlim(-1, 40)
plt.xticks(fontsize=9)
plt.xlabel('Duration (in minutes) for Caltrain Location trips')
plt.show()
display()

sanfrancisco = [37.7749, -122.4194]
def density_map(latitudes, longitudes, center=sanfrancisco, bins=1000, radius=0.1):
  cmap = copy.copy(plt.cm.jet)
  # Fill background with black
  cmap.set_bad((0,0,0))
  
  # Center the map around the provided center coordinates
  histogram_range = [
    [center[1] - radius, center[1] + radius],
    [center[0] - radius, center[0] + radius]
  ]
  
  plt.figure(figsize=(5,5)) 
  plt.hist2d(longitudes, latitudes, bins=bins, norm=LogNorm(), cmap=cmap, range=histogram_range) 

  # Remove all axes and annotations to keep the map clean and simple
  plt.grid('off')
  plt.axis('off')
  plt.gca().xaxis.set_visible(False)
  plt.gca().yaxis.set_visible(False)
  plt.tight_layout()
  plt.show()
  
  density_map(df['latitude'], df['longitude'])
  display()
