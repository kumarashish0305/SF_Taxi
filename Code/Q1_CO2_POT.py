import pyarrow.parquet as pq
import numpy as np
import pandas as pd

#Read the parquet dataframe and convert to pandas dataframe
df = sqlContext.read.parquet('/mnt/parquet2/sf_taxi_new')
df = df.toPandas()

Average_Waiting_Distance = df['Waiting_Distance'].mean() * 0.621371
print("Average distance travelled by cabs without passengers: " + str(Average_Waiting_Distance) + " miles")

#Source: https://sanfrancisco.cbslocal.com/2016/11/07/number-of-uber-lyft-drivers-dwarfs-san-franciscos-taxi-fleet/

  Total_cabs_2017 = 1800
  Agg_cars_2017 = 45000
  Average_CO2_emission = 404

  #Combustion_to_Electric_vehicle conversion rate
  Con_rate = 0.1

  #Total cars converted monthly
  #Let's assume max capacity of 250
  M_cars_conv = max(250, Con_rate*Total_cabs_2017)
  Total_agg_cabs_conv = max(250, Con_rate*(Total_cabs_2017 + Agg_cars_2017))

  #Emmission reduction potential 2017
  Monthly_CO2_Cabs = (Average_CO2_emission - 0) * M_cars_conv * Average_Waiting_Distance
  Monthly_CO2_Agg = (Average_CO2_emission - 0) * Total_agg_cabs_conv * Average_Waiting_Distance

  #Yearly Potential
  Pot_cabs_2017 = round((12 * Monthly_CO2_Cabs) / 1000, 2)
  Pot_agg_cabs_2017 = round((12 * Monthly_CO2_Agg) / 1000 , 2)
  
  #Print Result 1
  print("Yearly potential CO2 emission for taxi fleet: " + str(Pot_cabs_2017) + " Kilograms")
  #Yearly potential CO2 emission for taxi fleet: 1758.99 Kilograms

  
  #Print Result 2
  print("Yearly potential CO2 emission for entire car transportation fleet: " + str(Pot_agg_cabs_2017) + " Kilograms")
  #Yearly potential CO2 emission for entire car transportation fleet: 32928.2 Kilograms
